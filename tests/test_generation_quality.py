import os
import pytest
import torch
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer
from src.model import DecoderOnlyTransformer
from src.inference import TextGenerator

CHECKPOINT_PATH = "checkpoints/v3_best_model.pt"
VOCAB_PATH = "data/tokenizer/vocab_v3.json"

@pytest.fixture
def setup_generator():
    assert os.path.exists(CHECKPOINT_PATH), "v3_best_model.pt must exist."
    assert os.path.exists(VOCAB_PATH), "vocab_v3.json must exist."
    
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.load_vocab(VOCAB_PATH)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = DecoderOnlyTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    
    generator = TextGenerator(model, tokenizer, device="cpu")
    return generator

def test_generation_with_temperatures(setup_generator):
    """Verify that generation functions properly with various temperatures."""
    generator = setup_generator
    prompt = "kecerdasan"
    
    for temp in [0.7, 1.0, 1.2]:
        text, token_ids = generator.generate(
            prompt=prompt,
            max_new_tokens=10,
            temperature=temp,
            top_k=5,
            add_special_tokens=True
        )
        assert isinstance(text, str)
        assert len(token_ids) > 0

def test_generation_with_top_ks(setup_generator):
    """Verify that generation functions properly with various top-k settings."""
    generator = setup_generator
    prompt = "pytorch"
    
    for k in [5, 10, 20]:
        text, token_ids = generator.generate(
            prompt=prompt,
            max_new_tokens=10,
            temperature=1.0,
            top_k=k,
            add_special_tokens=True
        )
        assert isinstance(text, str)
        assert len(token_ids) > 0

def test_generation_max_tokens_limit(setup_generator):
    """Verify that generation length is strictly bounded by max_new_tokens."""
    generator = setup_generator
    tokenizer = generator.tokenizer
    prompt = "proses"
    
    max_new = 5
    text, token_ids = generator.generate(
        prompt=prompt,
        max_new_tokens=max_new,
        temperature=1.0,
        top_k=5,
        add_special_tokens=True
    )
    
    start_len = len(tokenizer.encode(prompt, add_special_tokens=True))
    assert len(token_ids) <= start_len + max_new
