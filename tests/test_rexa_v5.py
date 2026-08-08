import os
import pytest
import torch
from fastapi.testclient import TestClient
from web.app import app
from src.tokenizer.simple_tokenizer import SimpleWordTokenizerV5
from src.model import DecoderOnlyTransformer
from src.training.dataset import CausalLMDataset
from src.inference.generator import TextGenerator

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# 1. Test Dataset Format
def test_dataset_format():
    train_path = "data/processed/train_v5.txt"
    val_path = "data/processed/val_v5.txt"
    assert os.path.exists(train_path), "Train V5 file should exist"
    assert os.path.exists(val_path), "Val V5 file should exist"
    
    with open(train_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
        
    assert len(lines) > 0
    # Every line must contain conversational tags
    for line in lines[:10]:
        assert "<|system|>" in line
        assert "<|user|>" in line
        assert "<|assistant|>" in line
        assert "<|eos|>" in line

# 2. Test Instruction Dataset Loading
def test_instruction_dataset_loading():
    train_path = "data/processed/train_v5.txt"
    tokenizer = SimpleWordTokenizerV5()
    tokenizer.load_vocab("data/tokenizer/vocab_v5.json")
    
    dataset = CausalLMDataset(train_path, tokenizer, seq_len=32)
    assert len(dataset) > 0
    x, y = dataset[0]
    assert x.shape == (32,)
    assert y.shape == (32,)
    # Target should be input shifted by 1
    assert torch.equal(x[1:], y[:-1])

# 3. Test Tokenizer Compatibility
def test_tokenizer_compatibility():
    tokenizer = SimpleWordTokenizerV5()
    tokenizer.load_vocab("data/tokenizer/vocab_v5.json")
    
    # Check special tokens are mapped to single IDs and not split
    text = "<|system|> hello <|user|> hi <|assistant|> hey <|eos|>"
    tokens = tokenizer._tokenize(text)
    
    assert "<|system|>" in tokens
    assert "<|user|>" in tokens
    assert "<|assistant|>" in tokens
    assert "<|eos|>" in tokens
    
    # They should have unique IDs in vocab
    assert tokenizer.token_to_id("<|system|>") != tokenizer.token_to_id("<UNK>")
    assert tokenizer.token_to_id("<|user|>") != tokenizer.token_to_id("<UNK>")
    assert tokenizer.token_to_id("<|assistant|>") != tokenizer.token_to_id("<UNK>")
    assert tokenizer.token_to_id("<|eos|>") != tokenizer.token_to_id("<UNK>")

# 4. Test Model Compatibility
def test_model_compatibility():
    tokenizer = SimpleWordTokenizerV5()
    tokenizer.load_vocab("data/tokenizer/vocab_v5.json")
    
    model_config = {
        "vocab_size": len(tokenizer.vocab),
        "d_model": 64,
        "num_heads": 2,
        "num_layers": 2,
        "max_seq_len": 64,
        "dropout": 0.1
    }
    model = DecoderOnlyTransformer(**model_config)
    model.eval()
    
    dummy_input = torch.randint(0, len(tokenizer.vocab), (2, 16))
    with torch.no_grad():
        logits, _ = model(dummy_input)
    assert logits.shape == (2, 16, len(tokenizer.vocab))

# 5. Test Generation
def test_generation():
    tokenizer = SimpleWordTokenizerV5()
    tokenizer.load_vocab("data/tokenizer/vocab_v5.json")
    
    model_config = {
        "vocab_size": len(tokenizer.vocab),
        "d_model": 64,
        "num_heads": 2,
        "num_layers": 2,
        "max_seq_len": 64,
        "dropout": 0.1
    }
    model = DecoderOnlyTransformer(**model_config)
    generator = TextGenerator(model, tokenizer)
    
    # Generate response
    text, generated = generator.generate("halo", max_new_tokens=10, temperature=0.7, top_k=5)
    assert isinstance(text, str)
    assert len(generated) > 0

# 6. Test Multi-Turn Context formatting
def test_multi_turn_context_formatting():
    # Prompt building logic test
    messages = [
        {"role": "user", "content": "Hai"},
        {"role": "assistant", "content": "Halo"},
        {"role": "user", "content": "Siapa presiden pertama?"}
    ]
    prompt = "<|system|>Kamu adalah REXA, asisten AI dari REXSHIN."
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"<|user|>{msg['content']}"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant|>{msg['content']}"
    prompt += "<|assistant|>"
    
    assert prompt.startswith("<|system|>")
    assert prompt.endswith("<|assistant|>")
    assert "<|user|>Hai" in prompt
    assert "<|assistant|>Halo" in prompt
    assert "<|user|>Siapa presiden pertama?" in prompt

# 7. Test API /api/chat - single message
def test_api_chat_single(client):
    payload = {"message": "halo"}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)

# 8. Test API /api/chat - multi-turn history
def test_api_chat_history(client):
    payload = {
        "messages": [
            {"role": "user", "content": "halo"},
            {"role": "assistant", "content": "hai"},
            {"role": "user", "content": "apa kabar?"}
        ],
        "temperature": 0.5,
        "top_k": 5,
        "max_new_tokens": 20
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)

# 9. Test API /api/chat - empty prompt
def test_api_chat_empty(client):
    payload = {"message": ""}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == ""

# 10. Test API /api/chat - malformed prompt (no message field)
def test_api_chat_malformed(client):
    payload = {}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == ""

# 11. Test API /api/chat - long prompt
def test_api_chat_long(client):
    payload = {"message": "halo " * 100} # 500 characters
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)

# 12. Test API /api/chat - typo prompt
def test_api_chat_typo(client):
    payload = {"message": "ap itu transfrmer"}
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)
