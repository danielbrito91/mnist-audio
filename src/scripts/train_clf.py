import os
from pathlib import Path

from torch.utils.data import DataLoader

from src.mnist_audio.config import (
    DATA_RAW_DIR,
    SAMPLE_RATE,
    TEST_CACHE_PATH,
    TRAIN_CACHE_PATH,
)
from src.mnist_audio.data import (
    AudioMNISTDataset,
    create_split_from_files,
)
from src.mnist_audio.data.cache import ParquetAudioCache
from src.mnist_audio.models import SimpleCNN
from src.mnist_audio.preprocessing import STFTProcessor
from src.mnist_audio.training.clf_trainer import train_model

files = [str(f) for f in list(DATA_RAW_DIR.glob('**/*.wav'))]
train_files, test_files = create_split_from_files(files)

train_cache = ParquetAudioCache(Path(TRAIN_CACHE_PATH))
test_cache = ParquetAudioCache(Path(TEST_CACHE_PATH))

preprocessor = STFTProcessor(sample_rate=SAMPLE_RATE)

train_dataset = AudioMNISTDataset(train_files, train_cache, preprocessor)
test_dataset = AudioMNISTDataset(test_files, test_cache, preprocessor)


def main() -> None:
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=os.cpu_count() or 0,  # spawn-safe after pickling fix
        persistent_workers=True,
        pin_memory=False,  # MPS backend ignores pinned memory
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=os.cpu_count() or 0,
        persistent_workers=True,
        pin_memory=False,
    )

    # Create model
    model = SimpleCNN(num_classes=10)

    # Train model
    train_model(
        model,
        train_loader,
        val_loader,
        criterion=None,
        optimizer=None,
        device=None,
        num_epochs=100,
        lr=1e-3,
        save_dir=None,
    )


if __name__ == '__main__':
    main()
