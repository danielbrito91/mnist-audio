from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class AudioMetadata:
    file_path: str
    speaker_id: int
    utt_id: int
    label: int

    def to_dict(self) -> Dict[str, Any]:
        """Return the metadata as a plain ``dict`` helper."""
        return asdict(self)
