from dataclasses import dataclass
from pathlib import Path

DATA_RAW_DIR = Path('data/raw/data')
TRAIN_CACHE_PATH = Path('data/.cache/audio_mnist_train.zstd')
TEST_CACHE_PATH = Path('data/.cache/audio_mnist_test.zstd')
BEST_MODEL_PATH = Path('checkpoints/best_model.pth')
SAMPLE_RATE = 16000
SEED = 42
EXPECTED_COLUMNS = [
    'idx',
    'mel_flat',
    'n_mels',
    'n_frames',
    'file_path',
    'speaker_id',
    'utt_id',
    'label',
]


@dataclass
class STFTConfig:
    sample_rate: int = SAMPLE_RATE
    hop_length_ms: float = 12.5
    win_length_ms: float = 50
    n_fft: int = 2048
    window = 'hann'
    n_mels = 80
    preemphasis_coef = 0.97
    center = True
    power = 1
    pad_mode = 'reflect'
    target_duration_s = 0.65
