from pathlib import Path
from typing import Tuple, Union

import librosa
import torch

from src.mnist_audio.config import BEST_MODEL_PATH, SAMPLE_RATE
from src.mnist_audio.models import SimpleCNN
from src.mnist_audio.preprocessing import STFTProcessor
from src.mnist_audio.utils.torch_utils import get_device


def load_model(
    checkpoint_path: Union[str, Path] = BEST_MODEL_PATH,
    device: Union[torch.device, str, None] = None,
) -> SimpleCNN:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():  # Defensive guard-clause
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path.resolve()}'"
        )

    device_t = get_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device_t)

    model = SimpleCNN(num_classes=10).to(device_t)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def preprocess_audio(
    file_path: Union[str, Path], preprocessor: STFTProcessor
) -> torch.Tensor:
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    spec = preprocessor.transform(y)

    # (1, n_mels, n_frames) – add channel dim
    return torch.from_numpy(spec).float().unsqueeze(0)


def predict_tensor(
    model: SimpleCNN,
    mel_tensor: torch.Tensor,
    device: Union[torch.device, str, None] = None,
) -> Tuple[int, torch.Tensor]:
    device_t = get_device(device)
    # Add the batch dimension
    mel_tensor = mel_tensor.unsqueeze(0).to(device_t)

    with torch.no_grad():
        logits = model(mel_tensor)
        pred = int(torch.argmax(logits, dim=1).item())
    return pred, logits.squeeze(0).cpu()
