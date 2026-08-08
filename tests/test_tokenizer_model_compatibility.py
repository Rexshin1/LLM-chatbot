import os
import json
import pytest
import torch
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer
from src.model import DecoderOnlyTransformer
from src.inference import TextGenerator

VOCAB_V3_PATH = "data/tokenizer/vocab_v3.json"
CHECKPOINT_V3_PATH = "checkpoints/v3_best_model.pt"

@pytest.fixture
def setup_v3():
    assert os.path.exists(VOCAB_V3_PATH), "vocab_v3.json must exist."
    assert os.path.exists(CHECKPOINT_V3_PATH), "v3_best_model.pt must exist."
    
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.load_vocab(VOCAB_V3_PATH)
    
    checkpoint = torch.load(CHECKPOINT_V3_PATH, map_location="cpu")
    model = DecoderOnlyTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return {
        "tokenizer": tokenizer,
        "model": model,
        "checkpoint": checkpoint
    }

def test_vocab_size_match(setup_v3):
    """1. len(vocab_v3) cocok dengan model vocab_size"""
    tokenizer = setup_v3["tokenizer"]
    model = setup_v3["model"]
    checkpoint = setup_v3["checkpoint"]
    
    vocab_size_json = len(tokenizer.vocab)
    vocab_size_model = model.embeddings.tok_embed.num_embeddings
    vocab_size_config = checkpoint["model_config"]["vocab_size"]
    
    assert vocab_size_json == vocab_size_model
    assert vocab_size_json == vocab_size_config

def test_token_ids_in_range(setup_v3):
    """2. seluruh token ID vocabulary berada dalam range model [0, vocab_size - 1]"""
    tokenizer = setup_v3["tokenizer"]
    vocab_size = len(tokenizer.vocab)
    
    for token, idx in tokenizer.vocab.items():
        assert 0 <= idx < vocab_size, f"Token ID {idx} for '{token}' is out of range!"

def test_special_tokens_consistency(setup_v3):
    """3. seluruh special token ID konsisten"""
    tokenizer = setup_v3["tokenizer"]
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    
    for idx, token in enumerate(special_tokens):
        assert token in tokenizer.vocab
        assert tokenizer.vocab[token] == idx

def test_encode_valid_ids(setup_v3):
    """4. encode menghasilkan ID yang valid"""
    tokenizer = setup_v3["tokenizer"]
    vocab_size = len(tokenizer.vocab)
    
    text = "kecerdasan buatan dan machine learning."
    encoded = tokenizer.encode(text, add_special_tokens=True)
    
    for idx in encoded:
        assert 0 <= idx < vocab_size

def test_decode_valid_ids(setup_v3):
    """5. decode dapat menerjemahkan ID valid"""
    tokenizer = setup_v3["tokenizer"]
    
    ids = list(range(len(tokenizer.vocab)))
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    
    assert isinstance(decoded, str)
    assert len(decoded) > 0

def test_generated_ids_in_range(setup_v3):
    """6. generated ID selalu berada dalam range model"""
    model = setup_v3["model"]
    tokenizer = setup_v3["tokenizer"]
    generator = TextGenerator(model, tokenizer, device="cpu")
    
    vocab_size = len(tokenizer.vocab)
    prompt = "pytorch"
    
    _, generated_ids = generator.generate(
        prompt=prompt,
        max_new_tokens=15,
        temperature=1.0,
        top_k=5,
        add_special_tokens=True
    )
    
    for idx in generated_ids:
        assert 0 <= idx < vocab_size, f"Generated ID {idx} is out of range [0, {vocab_size-1}]!"

def test_checkpoint_vocab_compatibility(setup_v3):
    """7. checkpoint menggunakan vocabulary yang kompatibel"""
    checkpoint = setup_v3["checkpoint"]
    tokenizer = setup_v3["tokenizer"]
    
    state_dict = checkpoint["model_state_dict"]
    tok_embed_weight = state_dict["embeddings.tok_embed.weight"]
    lm_head_weight = state_dict["lm_head.weight"]
    
    assert tok_embed_weight.shape[0] == len(tokenizer.vocab)
    assert lm_head_weight.shape[0] == len(tokenizer.vocab)

def test_no_old_vocab_leak(setup_v3):
    """8. tidak ada vocabulary lama yang tidak sengaja digunakan"""
    tokenizer = setup_v3["tokenizer"]
    
    assert tokenizer.vocab.get("yang") == 180
    assert tokenizer.vocab.get("memungkinkan") != 244

def test_encode_model_decode_loop(setup_v3):
    """9. encode -> model -> decode menggunakan mapping yang sama"""
    tokenizer = setup_v3["tokenizer"]
    model = setup_v3["model"]
    
    prompt = "proses"
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    
    with torch.no_grad():
        logits, _ = model(input_tensor)
        
    next_token_id = torch.argmax(logits[0, -1, :]).item()
    
    assert 0 <= next_token_id < len(tokenizer.vocab)
    decoded_token = tokenizer.id_to_token_fn(next_token_id)
    assert isinstance(decoded_token, str)
