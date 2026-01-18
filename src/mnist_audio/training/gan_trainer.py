import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.mnist_audio.models.gan_model import Discriminator, Generator


def train_gan_epoch(
    generator: Generator,
    discriminator: Discriminator,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    n_mels: int,
    n_frames: int,
) -> None:
    generator.train()
    discriminator.train()
    running_generator_loss = 0.0
    running_discriminator_loss = 0.0
    pbar = tqdm(dataloader, desc='Training GAN')
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        z_dim = n_mels * n_frames
        z = torch.randn(data.shape[0], z_dim).to(device)

        fake_images = generator(z)
        real_images = data

        ones = torch.ones(data.shape[0], 1).to(device)
        zeros = torch.zeros(data.shape[0], 1).to(device)

        generator_loss = criterion(discriminator(fake_images), ones)
        discriminator_loss = (
            criterion(discriminator(real_images), ones)
            + criterion(discriminator(fake_images), zeros) / 2
        )

        generator_loss.backward()
        discriminator_loss.backward()

        generator_optimizer.step()
        discriminator_optimizer.step()

        running_generator_loss += generator_loss.item()
        running_discriminator_loss += discriminator_loss.item()

        pbar.set_postfix({
            'Generator Loss': f'{generator_loss.item():.4f}',
            'Discriminator Loss': f'{discriminator_loss.item():.4f}',
        })

    epoch_generator_loss = running_generator_loss / len(dataloader.dataset)
    epoch_discriminator_loss = running_discriminator_loss / len(
        dataloader.dataset
    )

    return {
        'generator_loss': epoch_generator_loss,
        'discriminator_loss': epoch_discriminator_loss,
    }


def train_gan_model(
    generator: Generator,
    discriminator: Discriminator,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    n_mels: int,
    n_frames: int,
    num_epochs: int,
) -> None:
    for epoch in range(num_epochs):
        train_gan_epoch(
            generator,
            discriminator,
            criterion,
            dataloader,
            device,
            generator_optimizer,
            discriminator_optimizer,
            n_mels,
            n_frames,
        )
