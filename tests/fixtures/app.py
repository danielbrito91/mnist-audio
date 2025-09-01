import pytest
from fastapi.testclient import TestClient

from src.mnist_audio.api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
