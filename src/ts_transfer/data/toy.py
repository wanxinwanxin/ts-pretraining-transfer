# Generate y = 2x + ε, ε ~ N(0, 0.1²), x ~ U(−1, 1), N=1000 — as a custom Dataset
# in src/ts_transfer/data/toy.py, seeded and reproducible.

import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, seed=42, n=1000, noise_std=0.1):
        self.generator = torch.Generator().manual_seed(seed)
        self.n = n
        self.noise_std = noise_std
        x = torch.rand(n, generator=self.generator) * 2 - 1
        self.x = x.unsqueeze(1)
        self.y = 2 * x + torch.randn(n, generator=self.generator) * self.noise_std
        self.y = self.y.unsqueeze(1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y
