# Generator

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(
        self, z_dim: int, hidden_dim: int, n_mels: int, n_frames: int
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_mels * n_frames),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, n_mels: int, n_frames: int, hidden_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_mels * n_frames, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
