from typing import List

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = Field('ok', description='Service heartbeat')


class PredictRequest(BaseModel):
    file_path: str = Field(..., description='Local path to the audio WAV file')

    @field_validator('file_path')
    @classmethod
    def _validate_wav(cls, v: str) -> str:
        if not v.lower().endswith('.wav'):
            raise ValueError('file_path must point to a .wav file')
        return v


class PredictResponse(BaseModel):
    digit: int = Field(..., ge=0, le=9)
    logits: List[float]
