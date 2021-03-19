import torch

class HMM(torch.nn.Module):
    """
    Hidden Markov Model.
    (For now, discrete observations only.)
    - forward(): computes the log probability of an observation sequence.
    - viterbi(): computes the most likely state sequence.
    - sample(): draws a sample from p(x).
    """
    def __init__(self, N, M, state_priors=None, transition_model=None, emission_model=None):
        super(HMM, self).__init__()
        self.N = N # number of states
        self.M = M # number of possible observations

        #self._unnormalized_state_priors = priors
        if state_priors is None:
            self.state_priors = StatePriors(N)
        else:
            self.state_priors = state_priors
        #self.unnormalized_state_priors = self.state_priors.unnormalized_state_priors

        if transition_model is None:
            self.transition_model = TransitionModel(self.N)
        else:
            self.transition_model = transition_model

        if emission_model is None:
            self.emission_model = EmissionModel(self.N, self.M)
        else:
            self.emission_model = emission_model

        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda:
            self.cuda()

    @classmethod
    def split_state(cls, old_hmm, split_idx):
        state_priors = StatePriors.split_state(old_hmm.state_priors, split_idx)
        transition_model = TransitionModel.split_state(old_hmm.transition_model, split_idx)
        emission_model = EmissionModel.split_state(old_hmm.emission_model, split_idx)
        return cls(old_hmm.N+1, old_hmm.M, state_priors, transition_model, emission_model)

    def forward(self, x, T):
        """
        x : IntTensor of shape (batch size, T_max)
        T : IntTensor of shape (batch size)

        Compute log p(x) for each example in the batch.
        T = length of each example
        """
        if self.is_cuda:
            x = x.cuda()
            T = T.cuda()

        batch_size = x.shape[0]; T_max = x.shape[1]
        log_state_priors = torch.nn.functional.log_softmax(self.state_priors.unnormalized_state_priors, dim=0)
        log_alpha = torch.zeros(batch_size, T_max, self.N)
        if self.is_cuda:
            log_alpha = log_alpha.cuda()

        log_alpha[:, 0, :] = self.emission_model(x[:,0]) + log_state_priors
        for t in range(1, T_max):
            log_alpha[:, t, :] = self.emission_model(x[:,t]) + self.transition_model(log_alpha[:, t-1, :], use_max=False)

        log_sums = log_alpha.logsumexp(dim=2)

        # Select the sum for the final timestep (each x has different length).
        log_probs = torch.gather(log_sums, 1, T.view(-1,1) - 1)
        return log_probs

    def sample(self, T=10):
        state_priors = torch.nn.functional.softmax(self.state_priors.unnormalized_state_priors, dim=0)
        transition_matrix = torch.nn.functional.softmax(self.transition_model.unnormalized_transition_matrix, dim=0)
        emission_matrix = torch.nn.functional.softmax(self.emission_model.unnormalized_emission_matrix, dim=1)

        # sample initial state
        z_t = torch.distributions.categorical.Categorical(state_priors).sample().item()
        z = []
        x = []
        z.append(z_t)
        for t in range(0,T):
            # sample emission
            x_t = torch.distributions.categorical.Categorical(emission_matrix[z_t]).sample().item()
            x.append(x_t)

            # sample transition
            z_t = torch.distributions.categorical.Categorical(transition_matrix[:,z_t]).sample().item()
            if t < T-1: z.append(z_t)

        return x, z

    def viterbi(self, x, T):
        """
        x : IntTensor of shape (batch size, T_max)
        T : IntTensor of shape (batch size)

        Find argmax_z log p(z|x) for each (x) in the batch.
        """
        if self.is_cuda:
            x = x.cuda()
            T = T.cuda()

        batch_size = x.shape[0]; T_max = x.shape[1]
        log_state_priors = torch.nn.functional.log_softmax(self.state_priors.unnormalized_state_priors, dim=0)
        log_delta = torch.zeros(batch_size, T_max, self.N).float()
        psi = torch.zeros(batch_size, T_max, self.N).long()
        if self.is_cuda:
            log_delta = log_delta.cuda()
            psi = psi.cuda()

        log_delta[:, 0, :] = self.emission_model(x[:,0]) + log_state_priors
        for t in range(1, T_max):
            max_val, argmax_val = self.transition_model(log_delta[:, t-1, :], use_max=True)
            log_delta[:, t, :] = self.emission_model(x[:,t]) + max_val
            psi[:, t, :] = argmax_val

        # Get the probability of the best path
        log_max = log_delta.max(dim=2)[0]
        best_path_scores = torch.gather(log_max, 1, T.view(-1,1) - 1)

        # This next part is a bit tricky to parallelize across the batch,
        # so we will do it separately for each example.
        z_star = []
        for i in range(0, batch_size):
            z_star_i = [ log_delta[i, T[i] - 1, :].max(dim=0)[1].item() ]
            for t in range(T[i] - 1, 0, -1):
                z_t = psi[i, t, z_star_i[0]].item()
                z_star_i.insert(0, z_t)

            z_star.append(z_star_i)

        return z_star, best_path_scores

def log_domain_matmul(log_A, log_B, use_max=False):
    """
    log_A : m x n
    log_B : n x p

    output : m x p matrix

    Normally, a matrix multiplication
    computes out_{i,j} = sum_k A_{i,k} x B_{k,j}

    A log domain matrix multiplication
    computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}

    This is needed for numerical stability
    when A and B are probability matrices.
    """
    m = log_A.shape[0]
    n = log_A.shape[1]
    p = log_B.shape[1]

    log_A_expanded = log_A.repeat(p, 1, 1).permute(1, 2, 0)
    log_B_expanded = log_B.repeat(m, 1, 1)

    elementwise_sum = log_A_expanded + log_B_expanded
    out = torch.logsumexp(elementwise_sum, dim=1)

    return out

def maxmul(log_A, log_B):
    m = log_A.shape[0]
    n = log_A.shape[1]
    p = log_B.shape[1]

    log_A_expanded = torch.stack([log_A] * p, dim=2)
    log_B_expanded = torch.stack([log_B] * m, dim=0)

    elementwise_sum = log_A_expanded + log_B_expanded
    out1, out2 = torch.max(elementwise_sum, dim=1)

    return out1, out2

class StatePriors(torch.nn.Module):
    def __init__(self, N, params=None):
        super(StatePriors, self).__init__()
        self.N = N

        if params is None:
            params = [torch.nn.Parameter(torch.randn(1)) for _ in range(N)]
        self._unnormalized_state_priors = torch.nn.ParameterList(params)

    @property
    def unnormalized_state_priors(self):
        return torch.cat([self._unnormalized_state_priors[i] for i in range(self.N)], dim=0)

    @classmethod
    def split_state(cls, old_model, i):
        new_params = []
        for j, param in enumerate(old_model._unnormalized_state_priors):
            pi = param.data
            g = False
            if j == i:
                g = True
                pi /= 2
            new_params.append(torch.nn.Parameter(pi, requires_grad=g))
        new_params.append(new_params[i])
        return cls(old_model.N+1, new_params)


class TransitionModel(torch.nn.Module):
    """
    - forward(): computes the log probability of a transition.
    - sample(): given a previous state, sample a new state.
    """
    def __init__(self, N, params=None):
        super(TransitionModel, self).__init__()
        self.N = N # number of states

        if params is None:
            params = [torch.nn.Parameter(torch.randn(N)) for _ in range(N)]

        self._unnormalized_transition_matrix = torch.nn.ParameterList(params)

    @property
    def unnormalized_transition_matrix(self):
        return torch.stack([self._unnormalized_transition_matrix[i] for i in range(self.N)], dim=1)

    @classmethod
    def split_state(cls, old_model, i):

        new_params = []
        for j, param in enumerate(old_model._unnormalized_transition_matrix):
            a_j = param.data
            if i == j:
                a_ij = a_j[i] / 2 
                a_j[i] = a_ij
            else:
                a_ij = a_j[i]
            new_params.append(torch.cat([a_j, a_ij.view(1)], dim=0))

        for p in range(len(new_params)):
            g = False
            if p == i or p == len(new_params) - 1:
                g = True
            new_params[p] = torch.nn.Parameter(new_params[p], requires_grad=g)

        new_params.append(new_params[i])
        return cls(old_model.N+1, new_params)

    def forward(self, log_alpha, use_max):
        """
        log_alpha : Tensor of shape (batch size, N)

        Multiply previous timestep's alphas by transition matrix (in log domain)
        """
        # Each col needs to add up to 1 (in probability domain)
        log_transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=0)

        # Matrix multiplication in the log domain
        if use_max:
            out1, out2 = maxmul(log_transition_matrix, log_alpha.transpose(0,1))
            return out1.transpose(0,1), out2.transpose(0,1)
        else:
            out = log_domain_matmul(log_transition_matrix, log_alpha.transpose(0,1))
            out = out.transpose(0,1)
            return out

class EmissionModel(torch.nn.Module):
    """
    - forward(): computes the log probability of an observation.
    - sample(): given a state, sample an observation for that state.
    """
    def __init__(self, N, M, params=None):
        super(EmissionModel, self).__init__()
        self.N = N # number of states
        self.M = M # number of possible observations

        if params is None:
            params = [torch.nn.Parameter(torch.randn(M)) for _ in range(N)]
        self._unnormalized_transition_matrix = torch.nn.ParameterList(params)

    @property
    def unnormalized_emission_matrix(self):
        return torch.stack([self._unnormalized_transition_matrix[i] for i in range(self.N)], dim=0)

    @property
    def entropy(self):
        m = self.unnormalized_emission_matrix.clone().detach()
        p = torch.softmax(m, dim=1)
        logp = torch.log(torch.clamp(p, 1e-12))
        e = -torch.sum(p * logp, dim=1)
        return e

    @classmethod
    def split_state(cls, old_model, i):
        new_params = []
        for k, param in enumerate(old_model._unnormalized_transition_matrix):
            g = False
            if k == i:
                g = True
            new_params.append(torch.nn.Parameter(param.data, requires_grad=g))
        new_params.append(new_params[i])
        return cls(old_model.N+1, old_model.M, new_params)

    def forward(self, x_t):
        """
        x_t : LongTensor of shape (batch size)

        Get observation probabilities
        """
        # Each col needs to add up to 1 (in probability domain)
        emission_matrix = torch.nn.functional.log_softmax(self.unnormalized_emission_matrix, dim=1)
        out = emission_matrix[:, x_t].transpose(0,1)

        return out
