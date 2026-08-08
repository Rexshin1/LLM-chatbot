import os
import pytest
from fastapi.testclient import TestClient
from web.app import app

@pytest.fixture(scope="module")
def client():
    # Using 'with' block triggers the startup events (startup_event) where model is loaded
    with TestClient(app) as c:
        yield c

def test_fastapi_import():
    """1. FastAPI app dapat diimport"""
    assert app is not None

def test_health_endpoint(client):
    """2. /api/health mengembalikan status 200"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] in ["V4", "V5"]
    assert data["device"] == "cpu"

def test_model_loaded(client):
    """3. model berhasil dimuat"""
    from web.app import model, tokenizer, generator
    assert model is not None
    assert tokenizer is not None
    assert generator is not None

def test_chat_endpoint_valid_request(client):
    """4. /api/chat menerima message dan 5. menghasilkan response string"""
    payload = {"message": "pytorch"}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)
