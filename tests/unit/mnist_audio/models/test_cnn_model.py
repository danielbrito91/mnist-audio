import torch

from src.mnist_audio.models.cnn_model import SimpleCNN


def test_forward_pass():
    model = SimpleCNN(num_classes=10)
    # batch_size=4, channels=1, n_mels=80, n_frames=53
    x = torch.randn(4, 1, 80, 53)
    output = model(x)
    assert output.shape == (4, 10)


def test_different_input_sizes():
    model = SimpleCNN(num_classes=5)
    for size in [28, 64, 128]:
        x = torch.randn(2, 1, size, size)
        output = model(x)
        assert output.shape == (2, 5)
