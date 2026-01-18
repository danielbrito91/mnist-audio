from pathlib import Path

from torch.utils.data import DataLoader

from src.mnist_audio.config import (
    DATA_RAW_DIR,
    TEST_CACHE_PATH,
    TRAIN_CACHE_PATH,
    STFTConfig,
)
from src.mnist_audio.data import (
    AudioMNISTDataset,
    create_split_from_files,
)
from src.mnist_audio.data.cache import ParquetAudioCache
from src.mnist_audio.models import Discriminator, Generator
from src.mnist_audio.preprocessing import STFTProcessor
from src.mnist_audio.training.gan_trainer import train_gan_model

files = [str(f) for f in list(DATA_RAW_DIR.glob('**/*.wav'))]
train_files, test_files = create_split_from_files(files)

train_cache = ParquetAudioCache(Path(TRAIN_CACHE_PATH))
test_cache = ParquetAudioCache(Path(TEST_CACHE_PATH))

preprocessor = STFTProcessor(config=STFTConfig())

train_dataset = AudioMNISTDataset(train_files, train_cache, preprocessor)
test_dataset = AudioMNISTDataset(test_files, test_cache, preprocessor)


def main() -> None:
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )

    # Create model
    z_dim, hidden_dim, n_mels, n_frames = 100, 128, 80, 53
    generator = Generator(
        z_dim=z_dim, hidden_dim=hidden_dim, n_mels=n_mels, n_frames=n_frames
    )
    discriminator = Discriminator(
        n_mels=n_mels, n_frames=n_frames, hidden_dim=hidden_dim
    )

    # Train model
    train_gan_model(
        generator,
        discriminator,
        train_loader,
        val_loader,
        criterion=None,
        device=None,
        num_epochs=100,
        lr_generator=2e-4,
        lr_discriminator=1e-4,
        save_dir=None,
        z_dim=z_dim,
        n_critic=1,  # Train D every batch
        n_gen=3,  # Train G 3x per D step (helps G catch up)
    )


if __name__ == '__main__':
    main()
