import pytest

from tests.factories import ParquetAudioCacheFactory

from .fixtures.app import client

pytest_plugins = [
    'tests.fixtures.model',  # exposes DummyModel, DummyPreprocessor & patches
]


@pytest.fixture
def cache_fixture(tmp_path_factory):
    return ParquetAudioCacheFactory(tmp_path_factory=tmp_path_factory)
