import torch
from tqdm import tqdm # for displaying progress bar
import os
import pandas as pd
from models import EmissionModel, TransitionModel, HMM
import numpy as np

class Trainer:
    def __init__(self, model, config, lr):
        self.model = model
        self.config = config
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.00001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.train_df = pd.DataFrame(columns=["loss","lr"])
        self.valid_df = pd.DataFrame(columns=["loss","lr"])

    def load_checkpoint(self, checkpoint_path):
        if os.path.isfile(os.path.join(checkpoint_path, "model_state.pth")):
            try:
                if self.model.is_cuda:
                    self.model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model_state.pth")))
                else:
                    self.model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model_state.pth"), map_location="cpu"))
            except:
                print("Could not load previous model; starting from scratch")
        else:
            print("No previous model; starting from scratch")

    def save_checkpoint(self, epoch, checkpoint_path):
        try:
            torch.save(self.model.state_dict(), os.path.join(checkpoint_path, "model_state.pth"))
        except:
            print("Could not save model")
        
    def train(self, dataset):
        train_loss = 0
        num_samples = 0
        split_every = 500
        print_interval = 100
        iterator = tqdm(dataset.loader, ncols=100)
        losses = []
        prev_loss = float("inf")
        print(f"num of states = {self.model.N}")
        self.model.train()
        for idx, batch in enumerate(iterator):
            x,T = batch
            batch_size = len(x)
            num_samples += batch_size
            log_probs = self.model(x,T)
            loss = -log_probs.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.cpu().data.numpy().item() * batch_size
            losses.append(loss.cpu().item())

                
            if idx % 1 == 0:
                iterator.set_description(
                    f"states = {self.model.N}; "
                    f"loss = {loss.cpu().item():.2f}"
                )

            if idx % print_interval == 0:
                
                print(f"\nloss = {loss.cpu().item():.2f}")
                for _ in range(5):
                    sampled_x, sampled_z = self.model.sample()
                    print("x = ", [self.config.vocab[s] for s in sampled_x])
                    print("z = ", sampled_z)

            if idx != 0 and idx % split_every == 0:
                curr_loss = np.mean(losses)
                #if curr_loss > prev_loss:
                #    self.model = self.old_model
                #    return train_loss
                self.old_model = self.model

                split_idx = int(torch.argmax(self.model.emission_model.entropy))
                print('\nsplit idx = ', split_idx)
                self.model = HMM.split_state(self.model, split_idx)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.00001)
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
                prev_loss = curr_loss
                losses = []

        train_loss /= num_samples
        return train_loss

    def test(self, dataset, print_interval=20):
        test_loss = 0
        num_samples = 0
        self.model.eval()
        print_interval = 1000
        for idx, batch in enumerate(dataset.loader):
            x,T = batch
            batch_size = len(x)
            num_samples += batch_size
            log_probs = self.model(x,T)
            loss = -log_probs.mean()
            test_loss += loss.cpu().data.numpy().item() * batch_size
            if idx % print_interval == 0:
                print(loss.item())
                sampled_x, sampled_z = self.model.sample()
                print("x = ", [self.config.vocab[s] for s in sampled_x])
                print("z = ", sampled_z)
        test_loss /= num_samples
        self.scheduler.step(test_loss) # if the validation loss hasn't decreased, lower the learning rate
        return test_loss
        
