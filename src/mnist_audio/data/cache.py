from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl

from src.mnist_audio.config import EXPECTED_COLUMNS
from src.mnist_audio.data.metadata import AudioMetadata


class ParquetAudioCache:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._df: pl.DataFrame | None = None

    def _read(self) -> pl.DataFrame:
        if self._df is None:
            self._df = pl.read_parquet(self.cache_path)
        return self._df

    def exists(self) -> bool:
        return self.cache_path.exists()

    def size(self) -> int:
        if not self.exists():
            return 0
        return self._read().shape[0]

    def load(self, idx: int) -> Tuple[np.ndarray, AudioMetadata]:
        if not self.exists():
            return None

        df = self._read()
        if idx < 0 or idx >= df.shape[0]:
            return None

        if not set(df.columns) == set(EXPECTED_COLUMNS):
            raise ValueError(
                f'Expected columns: {EXPECTED_COLUMNS}, but got: {df.columns}'
            )

        row = df.row(idx, named=True)
        mel_flat = np.array(row['mel_flat'])
        shape = (row['n_mels'], row['n_frames'])
        spectrogram = mel_flat.reshape(shape)
        metadata = AudioMetadata(
            file_path=row['file_path'],
            speaker_id=row['speaker_id'],
            utt_id=row['utt_id'],
            label=row['label'],
        )
        return spectrogram, metadata

    def load_all(self) -> List[Tuple[np.ndarray, AudioMetadata]]:
        if not self.exists():
            return []
        return [self.load(idx) for idx in range(self.size())]

    def save(
        self, spectrograms: List[np.ndarray], metadatas: List[AudioMetadata]
    ) -> None:
        data = []
        for idx, (spectrogram, metadata) in enumerate(
            zip(spectrograms, metadatas)
        ):
            row = {
                'idx': idx,
                'mel_flat': spectrogram.flatten().tolist(),
                'n_mels': spectrogram.shape[0],
                'n_frames': spectrogram.shape[1],
            }
            metadata_dict = metadata.to_dict()
            row.update(metadata_dict)
            data.append(row)

        cache_data = pl.DataFrame(data)
        cache_data.write_parquet(self.cache_path, compression='zstd')
