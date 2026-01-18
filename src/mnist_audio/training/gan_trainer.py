import json
from pathlib import Path
from time import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.mnist_audio.models.gan_model import Discriminator, Generator
from src.mnist_audio.utils.torch_utils import get_device


def save_checkpoint(
    generator: Generator,
    discriminator: Discriminator,
    epoch: int,
    generator_loss: float,
    discriminator_loss: float,
    path: Path,
) -> None:
    torch.save(
        {
            'epoch': epoch,
            'generator_loss': generator_loss,
            'discriminator_loss': discriminator_loss,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
        },
        path,
    )


def load_checkpoint(
    generator: Generator,
    discriminator: Discriminator,
    path: Path,
) -> None:
    checkpoint = torch.load(path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    return checkpoint


def train_gan_model(
    generator: Generator,
    discriminator: Discriminator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module | None,
    device: torch.device | None,
    num_epochs: int,
    lr_generator: float,
    lr_discriminator: float,
    save_dir: Path | None,
    z_dim: int,
    n_critic: int = 1,
    n_gen: int = 3,
) -> None:
    if device is None:
        device = get_device()

    if save_dir is None:
        save_dir = Path('checkpoints/gan')
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f'Using device: {device}')

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=lr_generator
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=lr_discriminator
    )

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    train_history = {'generator_loss': [], 'discriminator_loss': []}
    val_history = {'generator_loss': [], 'discriminator_loss': []}

    best_val_generator_loss = float('inf')

    best_model_path = save_dir / 'best_model.pth'

    print('Starting training...')

    for epoch in range(num_epochs):
        start_time = time()

        pbar = tqdm(train_loader, desc='Training GAN')
        running_generator_loss = 0.0
        running_discriminator_loss = 0.0
        for batch_idx, (X, _) in enumerate(pbar):
            X = X.to(device)
            batch_size = X.shape[0]
            z = torch.randn(batch_size, z_dim).to(device)

            # Label smoothing: 0.9 for real, 0.1 for fake
            # (prevents overconfidence)
            real_labels = torch.full((batch_size, 1), 0.9, device=device)
            fake_labels = torch.full((batch_size, 1), 0.1, device=device)

            # Add instance noise to inputs (decays over training)
            noise_std = max(0.1 * (1 - epoch / num_epochs), 0.0)
            X_noisy = X + noise_std * torch.randn_like(X)

            fake_images = generator(z)
            fake_noisy = fake_images + noise_std * torch.randn_like(
                fake_images
            )

            # Train Discriminator every n_critic batches
            if batch_idx % n_critic == 0:
                discriminator_optimizer.zero_grad()
                discriminator_loss = (
                    criterion(discriminator(X_noisy), real_labels)
                    + criterion(
                        discriminator(fake_noisy.detach()),
                        fake_labels,
                    )
                    / 2
                ) / 2
                discriminator_loss.backward()
                discriminator_optimizer.step()

            # Train Generator n_gen times per batch
            for _ in range(n_gen):
                z_gen = torch.randn(batch_size, z_dim).to(device)
                fake_for_g = generator(z_gen)
                generator_optimizer.zero_grad()
                generator_loss = criterion(
                    discriminator(fake_for_g), real_labels
                )
                generator_loss.backward()
                generator_optimizer.step()

            running_generator_loss += generator_loss.item()
            running_discriminator_loss += discriminator_loss.item()

            gen_loss = running_generator_loss / (batch_idx + 1)
            disc_loss = running_discriminator_loss / (batch_idx + 1)

            pbar.set_postfix({
                'Generator Loss': f'{gen_loss:.4f}',
                'Discriminator Loss': f'{disc_loss:.4f}',
            })

        batch_count = len(train_loader.dataset)
        gen_epoch_loss = running_generator_loss / batch_count
        disc_epoch_loss = running_discriminator_loss / batch_count

        train_history['generator_loss'].append(gen_epoch_loss)
        train_history['discriminator_loss'].append(disc_epoch_loss)

        # Run validation
        running_generator_val_loss = 0.0
        running_discriminator_val_loss = 0.0
        pbar = tqdm(val_loader, desc='Validating')
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            for batch_idx, (X, _) in enumerate(pbar):
                X = X.to(device)
                batch_size = X.shape[0]
                z = torch.randn(batch_size, z_dim).to(device)
                fake_images = generator(z)

                # Same label smoothing as training for consistent metrics
                real_labels = torch.full((batch_size, 1), 0.9, device=device)
                fake_labels = torch.full((batch_size, 1), 0.1, device=device)

                generator_loss = criterion(
                    discriminator(fake_images), real_labels
                )
                discriminator_loss = (
                    criterion(discriminator(X), real_labels)
                    + criterion(discriminator(fake_images), fake_labels)
                ) / 2

                running_generator_val_loss += generator_loss.item()
                running_discriminator_val_loss += discriminator_loss.item()

                gen_loss_val = running_generator_val_loss / (batch_idx + 1)
                disc_loss_val = running_discriminator_val_loss / (
                    batch_idx + 1
                )
                pbar.set_postfix({
                    'Generator Loss - Val': f'{gen_loss_val:.4f}',
                    'Discriminator Loss - Val': f'{disc_loss_val:.4f}',
                })
        generator.train()
        discriminator.train()

        epoch_generator_val_loss = running_generator_val_loss / len(
            val_loader.dataset
        )
        epoch_discriminator_val_loss = running_discriminator_val_loss / len(
            val_loader.dataset
        )

        val_history['generator_loss'].append(epoch_generator_val_loss)
        val_history['discriminator_loss'].append(epoch_discriminator_val_loss)

        if epoch_generator_val_loss < best_val_generator_loss:
            best_val_generator_loss = gen_loss_val
            save_checkpoint(
                generator,
                discriminator,
                epoch,
                gen_loss_val,
                disc_loss_val,
                best_model_path,
            )

        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            save_checkpoint(
                generator,
                discriminator,
                epoch,
                gen_loss_val,
                disc_loss_val,
                checkpoint_path,
            )

        epoch_time = time() - start_time

        print(f'Epoch {epoch + 1:3d}/{num_epochs}')
        print(f'Time: {epoch_time:.2f}s')
        print(
            f'Train: Generator Loss: {gen_loss:.4f},\
         Discriminator Loss: {disc_loss:.4f}'
        )
        print(
            f'Val: Generator Loss: {gen_loss_val:.4f},\
             Discriminator Loss: {disc_loss_val:.4f}'
        )

    with open(save_dir / 'gan_train_history.json', 'w') as f:
        json.dump(train_history, f)
    with open(save_dir / 'gan_val_history.json', 'w') as f:
        json.dump(val_history, f)

    return train_history, val_history
