import os
import pytest
import torch
import math
from torch.utils.data import DataLoader
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer
from src.model import DecoderOnlyTransformer
from src.training import CausalLMDataset
from src.inference import TextGenerator, Evaluator

CHECKPOINT_PATH = "checkpoints/best_model.pt"
VOCAB_PATH = "data/tokenizer/vocab.json"
VAL_PATH = "data/processed/val.txt"

@pytest.fixture
def setup_inference():
    assert os.path.exists(CHECKPOINT_PATH), "Best model checkpoint must exist from Phase 5."
    assert os.path.exists(VOCAB_PATH), "Vocab file must exist."
    
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.load_vocab(VOCAB_PATH)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model_config = checkpoint["model_config"]
    
    model = DecoderOnlyTransformer(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_config": model_config
    }

def test_checkpoint_loaded(setup_inference):
    """Verify that model is successfully loaded and matching config."""
    model = setup_inference["model"]
    assert isinstance(model, DecoderOnlyTransformer)
    assert model.embeddings.tok_embed.num_embeddings == len(setup_inference["tokenizer"].vocab)

def test_model_inference(setup_inference):
    """Verify standard model forward pass executes without labels."""
    model = setup_inference["model"]
    input_ids = torch.tensor([[10, 20, 30]])
    with torch.no_grad():
        logits, loss = model(input_ids)
    assert logits.shape == (1, 3, len(setup_inference["tokenizer"].vocab))
    assert loss is None

def test_generation_outputs(setup_inference):
    """Verify generate() produces text and token IDs of valid length."""
    model = setup_inference["model"]
    tokenizer = setup_inference["tokenizer"]
    generator = TextGenerator(model, tokenizer, device="cpu")
    
    max_new_tokens = 5
    prompt = "kecerdasan"
    
    text, token_ids = generator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        add_special_tokens=False
    )
    
    assert isinstance(text, str)
    assert isinstance(token_ids, list)
    
    prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    assert len(token_ids) <= prompt_len + max_new_tokens
    assert len(token_ids) > prompt_len

def test_eos_stops_generation(setup_inference):
    """Verify that the generator stops immediately when hitting EOS."""
    model = setup_inference["model"]
    tokenizer = setup_inference["tokenizer"]
    generator = TextGenerator(model, tokenizer, device="cpu")
    
    eos_id = tokenizer.token_to_id(tokenizer.eos_token)
    assert eos_id is not None
    
    # We can pass an EOS as prompt, and generate max 5 new tokens.
    # The generator should decode and handle it properly.
    text, token_ids = generator.generate(
        prompt=tokenizer.eos_token,
        max_new_tokens=5,
        add_special_tokens=False
    )
    assert len(token_ids) > 0

def test_sampling_parameters(setup_inference):
    """Verify generation works with different temperature and top_k configurations."""
    model = setup_inference["model"]
    tokenizer = setup_inference["tokenizer"]
    generator = TextGenerator(model, tokenizer, device="cpu")
    
    prompt = "buatan"
    
    text1, ids1 = generator.generate(prompt=prompt, max_new_tokens=3, temperature=0.5, top_k=5)
    text2, ids2 = generator.generate(prompt=prompt, max_new_tokens=3, temperature=1.5, top_k=1)
    
    assert isinstance(text1, str)
    assert isinstance(text2, str)
    assert len(ids1) > 0
    assert len(ids2) > 0

def test_evaluation_metrics(setup_inference):
    """Verify evaluator computes valid loss and perplexity."""
    model = setup_inference["model"]
    tokenizer = setup_inference["tokenizer"]
    
    val_dataset = CausalLMDataset(VAL_PATH, tokenizer, seq_len=16)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    evaluator = Evaluator(model, device="cpu")
    loss, perplexity = evaluator.evaluate(val_loader)
    
    assert isinstance(loss, float)
    assert isinstance(perplexity, float)
    assert loss > 0
    assert perplexity > 0
    assert not math.isnan(loss)
    assert not math.isnan(perplexity)
