import librosa
import numpy as np
import torch
import torchaudio

from src.mnist_audio.config import STFTConfig

from .vad import SileroVAD


class STFTProcessor:
    def __init__(
        self,
        config: STFTConfig,
    ):
        self.sample_rate = config.sample_rate
        self.hop_length = int(config.hop_length_ms * self.sample_rate / 1000)
        self.win_length = int(config.win_length_ms * self.sample_rate / 1000)
        self.n_fft = config.n_fft
        self.n_mels = config.n_mels
        self.window = config.window
        self.preemphasis_coef = config.preemphasis_coef
        self.center = config.center
        self.pad_mode = config.pad_mode
        self.power = config.power
        self.mel_basis = self._get_mel_basis()
        self.target_samples = int(config.target_duration_s * self.sample_rate)
        self.n_frames = 1 + self.target_samples // self.hop_length
        self.vad = SileroVAD(sampling_rate=self.sample_rate)

    def _preemphasis(self, x):
        return librosa.effects.preemphasis(x, coef=self.preemphasis_coef)

    def _get_mel_basis(self):
        return librosa.filters.mel(
            sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels
        )

    def _pad_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.shape[0] < self.target_samples:
            return np.pad(
                audio,
                (0, self.target_samples - audio.shape[0]),
                mode='constant',
                constant_values=0,
            )
        return audio

    def _trim_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.shape[0] > self.target_samples:
            return audio[: self.target_samples]
        return audio

    def transform(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio.squeeze()

        audio = self.vad.remove_leading_and_trailing_silence(
            torch.from_numpy(audio)
        ).numpy()
        # If the VAD trimmed everything, return a zero-filled
        # spectrogram with the expected shape
        # so downstream code can handle it uniformly.
        if audio.shape[0] == 0:
            return np.zeros((self.n_mels, self.n_frames))

        if audio.shape[0] > self.target_samples:
            audio = self._trim_audio(audio)
        else:
            audio = self._pad_audio(audio)

        S_mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            n_mels=self.n_mels,
            pad_mode=self.pad_mode,
            power=self.power,
        )

        return S_mel


def convert_from_spec_to_audio(
    S_linear, n_fft, hop_length, win_length, n_iter=32
):
    magnitude_spec = np.exp(S_linear)

    magnitude_spec = magnitude_spec**1.2

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window_fn=torch.hann_window,
        power=1,
        n_iter=n_iter,
    )

    y_torch = griffin_lim(torch.tensor(magnitude_spec))

    y_torch = y_torch.view(-1).cpu().numpy()
    return librosa.effects.deemphasis(y_torch, coef=0.97)
