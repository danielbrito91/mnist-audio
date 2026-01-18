"""DCGAN-style Generator and Discriminator for mel spectrograms."""

import torch
from torch import nn


class Generator(nn.Module):
    """Generator that maps z to mel spectrogram [1, n_mels, n_frames]."""

    def __init__(
        self, z_dim: int, hidden_dim: int, n_mels: int, n_frames: int
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames

        # Start with a small spatial size and upsample
        # Target: 80x53, start from 5x4 and upsample 4x
        # (5->10->20->40->80, 4->8->16->32->64)
        self.init_h = 5
        self.init_w = 4
        self.init_channels = hidden_dim * 8

        # Project latent to spatial tensor
        self.project = nn.Sequential(
            nn.Linear(z_dim, self.init_channels * self.init_h * self.init_w),
            nn.BatchNorm1d(self.init_channels * self.init_h * self.init_w),
            nn.ReLU(inplace=True),
        )

        # Upsample blocks: 5x4 -> 10x8 -> 20x16 -> 40x32 -> 80x64
        self.upsample = nn.Sequential(
            # 5x4 -> 10x8
            nn.ConvTranspose2d(
                hidden_dim * 8,
                hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            # 10x8 -> 20x16
            nn.ConvTranspose2d(
                hidden_dim * 4,
                hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            # 20x16 -> 40x32
            nn.ConvTranspose2d(
                hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # 40x32 -> 80x64
            nn.ConvTranspose2d(
                hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            # Final conv to get 1 channel, keep size
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [batch, z_dim]
        x = self.project(z)
        x = x.view(-1, self.init_channels, self.init_h, self.init_w)
        x = self.upsample(x)
        # Crop to exact target size (80x53 from 80x64)
        x = x[:, :, : self.n_mels, : self.n_frames]
        return x


class Discriminator(nn.Module):
    """Discriminator that classifies mel spectrograms as real or fake."""

    def __init__(self, n_mels: int, n_frames: int, hidden_dim: int):
        super().__init__()

        # Downsample blocks with strided convolutions
        self.features = nn.Sequential(
            # 80x53 -> 40x26
            nn.Conv2d(1, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 40x26 -> 20x13
            nn.Conv2d(
                hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # 20x13 -> 10x6
            nn.Conv2d(
                hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 10x6 -> 5x3
            nn.Conv2d(
                hidden_dim * 2,
                hidden_dim * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Calculate flattened size after convolutions
        # After 4 stride-2 convs: 80->40->20->10->5, 53->26->13->6->3
        self.flat_size = hidden_dim * 4 * 5 * 3

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, 1),
            # Note: Don't use Sigmoid here if using BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, n_mels, n_frames]
        features = self.features(x)
        return self.classifier(features)
