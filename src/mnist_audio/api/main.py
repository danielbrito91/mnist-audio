"""FastAPI entry-point for MNIST-Audio inference service."""

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from src.mnist_audio.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.mnist_audio.inference import predict_tensor, preprocess_audio

from .dependencies import T_ModelDep, T_PreprocessorDep

app = FastAPI(title='MNIST-Audio API', version='0.1.0')


@app.get('/health', response_model=HealthResponse)
async def health() -> HealthResponse:
    """Simple liveness probe."""
    return HealthResponse()


@app.post('/predict', response_model=PredictResponse, tags=['inference'])
async def predict(
    payload: PredictRequest, model: T_ModelDep, preprocessor: T_PreprocessorDep
) -> PredictResponse:  # type: ignore[arg-type]
    """Predict a single digit from an audio file path."""
    mel_tensor = await run_in_threadpool(
        preprocess_audio, payload.file_path, preprocessor
    )
    digit, logits_t = await run_in_threadpool(
        predict_tensor, model, mel_tensor
    )
    return PredictResponse(digit=digit, logits=logits_t.tolist())
