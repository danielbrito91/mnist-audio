"""FastAPI dependencies for shared resources (model, configs, etc.)."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.mnist_audio.config import STFTConfig
from src.mnist_audio.inference.predictor import load_model
from src.mnist_audio.models import SimpleCNN
from src.mnist_audio.preprocessing import STFTProcessor


@lru_cache(maxsize=1)
def _model_factory() -> SimpleCNN:  # pragma: no cover
    return load_model()


@lru_cache(maxsize=1)
def _preprocessor_factory() -> STFTProcessor:  # pragma: no cover
    return STFTProcessor(config=STFTConfig())


def get_model() -> SimpleCNN:  # pragma: no cover
    return _model_factory()


def get_preprocessor() -> STFTProcessor:  # pragma: no cover
    return _preprocessor_factory()


T_ModelDep = Annotated[SimpleCNN, Depends(get_model)]
T_PreprocessorDep = Annotated[STFTProcessor, Depends(get_preprocessor)]
