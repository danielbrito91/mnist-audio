import torch
from silero_vad import get_speech_timestamps, load_silero_vad


class SileroVAD:
    def __init__(self, sampling_rate: int = 16000):
        self.sampling_rate = sampling_rate
        if self.sampling_rate != 16000:
            raise ValueError('Sampling rate must be 16000 or 8000')

        self.vad_model = load_silero_vad()

    def _get_speech_timestamps(self, wav: torch.Tensor) -> torch.Tensor:
        return get_speech_timestamps(
            wav,
            self.vad_model,
            sampling_rate=self.sampling_rate,
            return_seconds=False,
        )

    def remove_leading_and_trailing_silence(
        self, wav: torch.Tensor
    ) -> torch.Tensor:
        if wav.numel() == 0:
            return wav

        speech_timestamps = self._get_speech_timestamps(wav)

        if not speech_timestamps:
            return torch.tensor([], dtype=wav.dtype, device=wav.device)

        start_sample = speech_timestamps[0]['start']
        end_sample = speech_timestamps[-1]['end']
        return wav[start_sample:end_sample]
