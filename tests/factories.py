from itertools import islice
from pathlib import Path

import factory
import numpy as np
import polars as pl
import pytest

from src.mnist_audio.config import EXPECTED_COLUMNS
from src.mnist_audio.data import AudioMetadata
from src.mnist_audio.data.cache import ParquetAudioCache

_N_MELS: int = 80
_N_FRAMES: int = 53


class AudioMetadataFactory(factory.Factory):
    """Factory for :class:`~src.mnist_audio.data.dataset.AudioMetadata`."""

    class Meta:
        model = AudioMetadata

    idx: int = factory.Sequence(lambda n: n)

    @classmethod
    def _create(cls, model_class, idx: int, **kwargs):
        speaker_id = 50 + idx
        utt_id = 40 + idx
        label = idx % 10
        file_path = f'data/raw/{speaker_id}/0_{speaker_id}_{utt_id}.wav'
        return model_class(
            file_path=file_path,
            speaker_id=speaker_id,
            utt_id=utt_id,
            label=label,
            **kwargs,
        )


class ParquetAudioCacheFactory(factory.Factory):
    """Factory for :class:`~src.mnist_audio.data.cache.ParquetAudioCache`."""

    class Meta:
        model = ParquetAudioCache

    rows: int = 5
    cache_path: Path | None = None

    @classmethod
    def _create(
        cls,
        model_class,
        rows: int,
        cache_path: Path | str | None = None,
        tmp_path_factory: pytest.TempPathFactory | None = None,
        **kwargs,
    ):
        if cache_path is None:
            if tmp_path_factory is None:
                raise ValueError(
                    'Either `cache_path` or `tmp_path_factory` must '
                    'be provided to ParquetAudioCacheFactory.'
                )
            cache_path = (
                tmp_path_factory.mktemp('cache') / 'audio_mnist_test.zstd'
            )
        cache_path = Path(cache_path)

        metadatas = list(islice(AudioMetadataFactory.create_batch(rows), rows))
        spectrograms = [
            np.random.randn(_N_MELS, _N_FRAMES).astype(np.float32)
            for _ in range(rows)
        ]

        parquet_rows = []
        for idx, (metadata, spec) in enumerate(zip(metadatas, spectrograms)):
            row = {
                'idx': idx,
                'mel_flat': spec.flatten().tolist(),
                'n_mels': _N_MELS,
                'n_frames': _N_FRAMES,
                **metadata.to_dict(),
            }
            parquet_rows.append(row)

        df = pl.DataFrame(parquet_rows).select(EXPECTED_COLUMNS)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path, compression='zstd')

        return ParquetAudioCache(cache_path)
