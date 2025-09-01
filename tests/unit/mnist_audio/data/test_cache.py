import numpy as np
import polars as pl
import pytest

from tests.factories import AudioMetadataFactory


def test_cache_load(cache_fixture):
    assert cache_fixture.exists()
    assert cache_fixture.size() == 5
    assert cache_fixture.load(0) is not None
    assert cache_fixture.load(99) is None
    assert cache_fixture.load_all() is not None
    assert isinstance(cache_fixture._read(), pl.DataFrame)


def test_cache_save(cache_fixture):
    cache_fixture.save(
        [np.random.randn(80, 53).astype(np.float32) for _ in range(5)],
        [AudioMetadataFactory() for _ in range(5)],
    )
    assert cache_fixture.exists()
    assert cache_fixture.size() == 5
    assert cache_fixture.load(0) is not None
    assert cache_fixture.load_all() is not None


def test_cache_raise_error_if_columns_do_not_match(cache_fixture):
    cache_fixture._read()
    # Patch the cache to have a different column name
    cache_fixture._df = cache_fixture._df.with_columns(
        pl.col('mel_flat').alias('mel_flat_2')
    )
    with pytest.raises(ValueError, match='Expected columns: '):
        cache_fixture.load(0)


def test_cache_does_not_exist(cache_fixture):
    cache_fixture.cache_path.unlink()
    assert not cache_fixture.exists()
    assert cache_fixture.load(0) is None
    assert cache_fixture.load_all() == []
