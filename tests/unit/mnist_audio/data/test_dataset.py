import numpy as np
import pytest
import torch

from src.mnist_audio.data.dataset import (
    AudioMNISTDataset,
    create_split_from_files,
)


@pytest.fixture
def audio_dataset_fixture(cache_fixture, preprocessor_fixture):
    files = cache_fixture._read().select('file_path').to_series().to_list()
    return AudioMNISTDataset(
        files=files,
        cache=cache_fixture,
        preprocessor=preprocessor_fixture,
    )


def test_audio_dataset_empty_files(cache_fixture, preprocessor_fixture):
    with pytest.raises(ValueError, match='No files provided'):
        AudioMNISTDataset(
            files=[],
            cache=cache_fixture,
            preprocessor=preprocessor_fixture,
        )


def test_get_audio_metadata(audio_dataset_fixture):
    metadata = audio_dataset_fixture._get_metadata(
        audio_dataset_fixture.files[0]
    )
    assert metadata.file_path == audio_dataset_fixture.files[0]
    assert isinstance(metadata.speaker_id, int)
    assert isinstance(metadata.utt_id, int)
    assert isinstance(metadata.label, int)
    assert metadata.label >= 0
    assert metadata.label < 10


def test_process_and_cache(audio_dataset_fixture):
    # Drop the cache
    original_cache = audio_dataset_fixture.cache._read()
    audio_dataset_fixture.cache.cache_path.unlink()
    assert not audio_dataset_fixture.cache.exists()

    # Process and cache
    audio_dataset_fixture._process_and_cache()
    assert audio_dataset_fixture.cache.exists()
    assert audio_dataset_fixture.cache.size() == 5
    assert audio_dataset_fixture.cache.load(0) is not None
    assert audio_dataset_fixture.cache.load_all() is not None

    # Check that the cache is the same as the original
    new_cache = audio_dataset_fixture.cache._read()

    assert original_cache.equals(new_cache)


@pytest.mark.skip(reason='This test is flaky')
def test_process_and_cache_silent_files(audio_dataset_fixture):
    # Mock y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    audio_dataset_fixture.preprocessor.transform = lambda x: np.zeros((80, 53))

    # Remove the cache
    audio_dataset_fixture.cache.cache_path.unlink()
    assert not audio_dataset_fixture.cache.exists()

    # Process and cache
    audio_dataset_fixture._process_and_cache()
    assert audio_dataset_fixture.files == []

    assert audio_dataset_fixture.cache.size() == 0


def test_get_state(audio_dataset_fixture):
    # Drop the un-picklable TorchScript model
    state = audio_dataset_fixture.__getstate__()
    assert state['preprocessor'] is None


def test_len(audio_dataset_fixture):
    assert len(audio_dataset_fixture) == 5


def test_getitem(audio_dataset_fixture):
    item = audio_dataset_fixture[0]
    assert isinstance(item, tuple)
    assert len(item) == 2
    assert isinstance(item[0], torch.Tensor)
    assert isinstance(item[1], torch.Tensor)


def test_split_from_files(audio_dataset_fixture):
    files = audio_dataset_fixture.files
    train_files, val_files = create_split_from_files(files, train_ratio=0.8)
    assert len(train_files) == 4
    assert len(val_files) == 1
