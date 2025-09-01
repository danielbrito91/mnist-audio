import pytest
import torch

from src.mnist_audio.inference.predictor import (
    load_model,
    predict_tensor,
    preprocess_audio,
)


def test_preprocess_audio(preprocessor_fixture):
    file_path = 'fake_file.wav'
    tensor = preprocess_audio(file_path, preprocessor_fixture)
    assert tensor.shape == (1, 80, 53)


@pytest.mark.parametrize('device', ['cpu', None])
def test_predict_tensor(model_fixture, device):
    tensor = torch.randn(1, 80, 53)
    pred, logits = predict_tensor(model_fixture, tensor, device=device)
    assert pred in range(10)
    assert logits.shape == (10,)


def test_load_model_with_checkpoint(model_fixture):
    model = load_model(model_fixture)
    assert model is not None