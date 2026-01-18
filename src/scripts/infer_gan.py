from pathlib import Path

import librosa
import numpy as np
import torch

from src.mnist_audio.data import db_to_power, denormalize_mel
from src.mnist_audio.models.gan_model import Discriminator, Generator
from src.mnist_audio.training.gan_trainer import load_checkpoint

SAMPLE_RATE = 16000
HOP_LENGTH = int(12.5 * SAMPLE_RATE / 1000)  # 200
WIN_LENGTH = int(50 * SAMPLE_RATE / 1000)  # 800
N_FFT = 2048


def load_gan(
    gan_path: Path = Path('checkpoints/gan/best_model.pth'),
) -> Generator:
    generator = Generator(z_dim=100, hidden_dim=128, n_mels=80, n_frames=53)
    discriminator = Discriminator(n_mels=80, n_frames=53, hidden_dim=128)

    # load_checkpoint already loads state dicts into the models
    load_checkpoint(generator, discriminator, gan_path)

    generator.eval()
    del discriminator
    return generator


def infer_gan_mel(generator: Generator, z_dim: int = 100) -> np.ndarray:
    z = torch.randn(1, z_dim)
    fake_mels = generator(z)
    # Reshape generated mel to (n_mels, n_frames)
    fake_mel_2d = fake_mels.view(80, 53).detach().cpu().numpy()

    # Denormalize: [0, 1] -> [-80, 0] dB -> power mel
    mel_db = denormalize_mel(fake_mel_2d)
    mel_power = db_to_power(mel_db)

    return mel_power


def infer_gan_audio(mel_power: np.ndarray) -> np.ndarray:
    # librosa.feature.inverse.mel_to_audio inverts the mel spectrogram
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_iter=32,  # Griffin-Lim iterations
    )

    return audio
