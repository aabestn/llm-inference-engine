import pytest
from fastapi.testclient import TestClient
from llm_engine.api.server import app

client = TestClient(app)

def test_generate_endpoint():
    """Tests payload validation and response structure for non-streaming generation endpoint."""
    payload = {
        "prompt": "Explain continuous batching",
        "max_tokens": 64,
        "temperature": 0.7,
        "stream": False
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    assert data["finish_reason"] == "stop"

def test_empty_prompt_validation():
    """Ensures 400 status error when an empty prompt is supplied."""
    payload = {"prompt": "", "max_tokens": 10}
    response = client.post("/generate", json=payload)
    assert response.status_code == 400