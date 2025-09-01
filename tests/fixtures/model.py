import numpy as np
import pytest
import torch

from src.mnist_audio.api.dependencies import get_model, get_preprocessor
from src.mnist_audio.api.main import app
from src.mnist_audio.config import STFTConfig


class DummyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # Shape: (batch_size, 10) – one logit per digit
        return torch.zeros(batch_size, 10, device=x.device)


class DummyPreprocessor:
    def __init__(self) -> None:
        cfg = STFTConfig()
        self.n_mels: int = cfg.n_mels
        hop_length = int(cfg.hop_length_ms * cfg.sample_rate / 1000)
        self.n_frames: int = (
            1 + int(cfg.target_duration_s * cfg.sample_rate) // hop_length
        )

    def transform(self, audio):
        """Return a zero-filled spectrogram with the expected shape."""
        return np.zeros((self.n_mels, self.n_frames), dtype=np.float32)


@pytest.fixture(scope='session')
def model_fixture() -> DummyModel:
    return DummyModel()


@pytest.fixture(scope='session')
def preprocessor_fixture() -> DummyPreprocessor:
    return DummyPreprocessor()


@pytest.fixture(autouse=True)
def _override_fastapi_dependencies(
    model_fixture: DummyModel, preprocessor_fixture: DummyPreprocessor
):
    app.dependency_overrides[get_model] = lambda: model_fixture  # type: ignore[return-value]
    app.dependency_overrides[get_preprocessor] = lambda: preprocessor_fixture  # type: ignore[return-value]
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _patch_librosa_load():
    """Monkey-patch ``librosa.load`` to avoid reading real audio files."""

    import librosa

    original_load = librosa.load

    def _fake_load(path: str, sr: int | None = None):  # noqa: D401, ANN001
        sr_fallback = sr or 16_000
        return np.zeros(sr_fallback, dtype=np.float32), sr_fallback

    librosa.load = _fake_load
    yield
    librosa.load = original_load
