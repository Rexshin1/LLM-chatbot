import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from web.app import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

# 1. Test Gemini configuration (missing API key)
def test_gemini_missing_api_key(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    payload = {
        "message": "hello",
        "model": "gemini"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "Gemini API key is not configured."

# 2. Test Gemini API selection and success response with mock
@patch("google.genai.Client")
def test_gemini_success(mock_client_class, client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    
    # Mocking Client instances
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is a response from REXA Gemini."
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client
    
    payload = {
        "message": "hello",
        "model": "gemini",
        "history": [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a language."}
        ]
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "This is a response from REXA Gemini."
    assert data["model"] == "gemini"

# 3. Test Gemini API connection/error response
@patch("google.genai.Client")
def test_gemini_error(mock_client_class, client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API connection timed out")
    mock_client_class.return_value = mock_client
    
    payload = {
        "message": "hello",
        "model": "gemini"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 500
    data = response.json()
    assert data["error"] == "REXA could not connect to Gemini."

# 4. Test Local model selection fallback
def test_local_model_selection(client):
    payload = {
        "message": "15 dikali 3",
        "model": "local"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "local"
    assert "response" in data

# 5. Test health endpoint values
def test_health_endpoint(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["gemini_configured"] is True
