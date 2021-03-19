import torch
import torch.utils.data
from collections import Counter
import os
from sklearn.model_selection import train_test_split

class Config:
    def __init__(self,N,path):
        self.N = N
        self.path = path

def read_config(N, path):
    config = Config(N=N,path=path)
    return config

def get_datasets(config, parent_dir):
    lines = []

    vocab = set()
    data = {"train": [], "dev": []}
    for split in ["train", "dev"]:
        path = os.path.join(parent_dir, f"{split}.txt")

        with open(path, 'r') as f:
            for l in f:
                l = [int(ll) for ll in l.strip().split(',')]
                data[split].append(l)
                vocab.update(set(l))

    vocab = {v: i for i, v in enumerate(vocab)}

    # get input and output alphabets
    train_dataset = TextDataset(data["train"], vocab)
    valid_dataset = TextDataset(data["dev"], vocab)

    config.M = len(vocab)
    config.vocab = list(vocab)

    return train_dataset, valid_dataset


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, lines, vocab):
        self.lines = lines # list of strings
        self.vocab = vocab
        pad_and_one_hot = PadAndOneHot(vocab) # function for generating a minibatch from strings
        self.loader = torch.utils.data.DataLoader(self, batch_size=1024, num_workers=16, shuffle=True, collate_fn=pad_and_one_hot, pin_memory=True)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]


class PadAndOneHot:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, batch):
        """
        Returns a minibatch of strings, one-hot encoded and padded to have the same length.
        """
        xs = []
        x_lens = []
        batch_size = len(batch)
        for i in range(batch_size):
            x = batch[i]
            x = [self.vocab[xx] for xx in x]
            xs.append(x)
            x_lens.append(len(x))

        # pad all sequences with 0 to have same length
        T = max(x_lens)
        for i in range(batch_size):
            xs[i] += [0] * (T - x_lens[i])
            xs[i] = torch.tensor(xs[i])

        xs = torch.stack(xs)
        x_lens = torch.tensor(x_lens)

        return (xs, x_lens)
