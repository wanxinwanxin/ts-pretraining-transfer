# Generate y = 2x + ε, ε ~ N(0, 0.1²), x ~ U(−1, 1), N=1000 — as a custom Dataset
# in src/ts_transfer/data/toy.py, seeded and reproducible.

import torch
import torch.nn as nn
from torch.utils.data import Dataset



class custom_toy_dataset(Dataset):
    def __init__(self, seed=42, n=1000, noise_std=0.1):
        self.generator = torch.Generator()
        self.seed = seed
        self.n = n
        self.noise_std = noise_std

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self.generator.manual_seed(idx+self.seed)
        x = torch.rand(1, generator=self.generator, requires_grad=False) * 2 - 1
        y = 2 * x + torch.randn(1, generator=self.generator) * self.noise_std
        return x, y