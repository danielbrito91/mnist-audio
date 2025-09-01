import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import librosa
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.mnist_audio.config import SAMPLE_RATE, SEED
from src.mnist_audio.preprocessing import STFTProcessor

from .cache import ParquetAudioCache
from .metadata import AudioMetadata


class AudioMNISTDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        cache: ParquetAudioCache,
        preprocessor: STFTProcessor,
    ):
        if len(files) == 0:
            raise ValueError('No files provided')

        # Keep an internal list of files that actually contain audio
        # (i.e. not pure silence). This list may shrink after preprocessing.
        self.files = files
        self.cache = cache
        self.preprocessor = preprocessor

        if not cache.exists() or cache.size() == 0:
            self._process_and_cache()

        self.size = cache.size()

    def _get_metadata(self, file_path: str) -> AudioMetadata:
        label, speaker_id, utt_id = Path(file_path).stem.split('_')
        return AudioMetadata(
            file_path=file_path,
            speaker_id=int(speaker_id),
            utt_id=int(utt_id),
            label=int(label),
        )

    def _process_and_cache(self) -> None:
        """Transform raw audio files, skipping those that are pure silence."""

        import numpy as np

        spectrograms: list = []
        metadatas: list = []
        valid_files: list = []

        for file_path in tqdm(
            self.files, desc=f'Processing files for {self.cache.cache_path}'
        ):
            y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
            spectrogram = self.preprocessor.transform(y)

            # Skip silent utterances (all zeros after VAD trimming)
            if not np.any(spectrogram):
                continue

            metadata = self._get_metadata(file_path)

            spectrograms.append(spectrogram)
            metadatas.append(metadata)
            valid_files.append(file_path)

            del y, spectrogram, metadata

        # Update internal file list to reflect only valid examples
        self.files = valid_files

        self.cache.save(spectrograms, metadatas)

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, AudioMetadata]:
        spectrogram, metadata = self.cache.load(idx)

        mel_tensor = torch.from_numpy(spectrogram).float().unsqueeze(0)
        label_tensor = torch.tensor(metadata.label, dtype=torch.long)

        return mel_tensor, label_tensor

    def __getstate__(self):
        # Drop the un-picklable TorchScript model
        state = self.__dict__.copy()
        state['preprocessor'] = None
        return state


def create_split_from_files(
    files: List[str], train_ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
    if train_ratio < 0 or train_ratio > 1:
        raise ValueError('train_ratio must be between 0 and 1')

    speaker_to_files = defaultdict(list)
    for file in files:
        speaker_id = int(Path(file).stem.split('_')[1])
        speaker_to_files[speaker_id].append(file)
    all_speakers = list(speaker_to_files.keys())
    n_test_speakers = max(1, int(len(speaker_to_files) * (1 - train_ratio)))

    random.seed(SEED)
    test_speakers = set(random.sample(all_speakers, n_test_speakers))
    train_speakers = set(all_speakers) - test_speakers

    train_files = []
    val_files = []
    for file in files:
        speaker_id = int(Path(file).stem.split('_')[1])
        if speaker_id in train_speakers:
            train_files.append(file)
        else:
            val_files.append(file)

    return train_files, val_files
