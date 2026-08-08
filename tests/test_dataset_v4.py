import os
import json
import pytest
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer

def test_dataset_v4_files_exist():
    """Verify that all raw, processed, and vocabulary files for V4 exist."""
    assert os.path.exists("data/raw/corpus_v4.txt")
    assert os.path.exists("data/processed/train_v4.txt")
    assert os.path.exists("data/processed/val_v4.txt")
    assert os.path.exists("data/tokenizer/vocab_v4.json")

def test_dataset_v4_no_leakage():
    """Verify that there is no data overlap/leakage between train and validation splits in V4."""
    with open("data/processed/train_v4.txt", "r", encoding="utf-8") as f:
        train_lines = set(line.strip() for line in f if line.strip())
        
    with open("data/processed/val_v4.txt", "r", encoding="utf-8") as f:
        val_lines = set(line.strip() for line in f if line.strip())
        
    overlap = train_lines.intersection(val_lines)
    assert len(overlap) == 0, f"Leakage detected! Overlapping lines: {overlap}"

def test_dataset_v4_vocab_coverage():
    """Verify that vocab_v4 covers all tokens in the validation split (0.0% UNK rate)."""
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.load_vocab("data/tokenizer/vocab_v4.json")
    
    with open("data/processed/val_v4.txt", "r", encoding="utf-8") as f:
        val_text = f.read()
        
    val_ids = tokenizer.encode(val_text, add_special_tokens=False)
    unk_id = tokenizer.token_to_id(tokenizer.unk_token)
    
    num_unks = val_ids.count(unk_id)
    unk_rate = num_unks / len(val_ids) if len(val_ids) > 0 else 0.0
    
    assert unk_rate == 0.0, f"UNK rate in validation split is {unk_rate:.4%}, expected 0.0%!"
