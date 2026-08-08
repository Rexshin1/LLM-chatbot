import os
import pytest
from src.dataset.preprocess import clean_text, normalize_text, split_data, preprocess_pipeline
from src.dataset.loader import TextDataset, get_dataloader

RAW_FILE = "data/raw/corpus.txt"
PROCESSED_DIR = "data/processed"

def test_dataset_files_exist():
    """Verify raw dataset is present in the raw directory."""
    assert os.path.exists(RAW_FILE), f"Raw dataset does not exist at {RAW_FILE}"

def test_text_cleaning():
    """Verify that clean_text removes extra whitespaces and empty lines."""
    dirty_text = "Hello    World! \n\n  This is  a test.  \t "
    cleaned = clean_text(dirty_text)
    assert "  " not in cleaned
    assert "\t" not in cleaned
    lines = cleaned.split('\n')
    assert len(lines) == 2
    assert lines[0] == "Hello World!"
    assert lines[1] == "This is a test."

def test_text_normalization():
    """Verify that normalize_text lowercases correctly."""
    mixed_text = "HELLO World! 123"
    normalized = normalize_text(mixed_text, lowercase=True)
    assert normalized == "hello world! 123"

def test_train_val_split():
    """Verify that splitting logic splits by lines correctly based on ratio."""
    dummy_text = "\n".join([f"Line {i}" for i in range(10)])
    train, val = split_data(dummy_text, val_split=0.2)
    
    train_lines = train.split('\n')
    val_lines = val.split('\n')
    
    assert len(train_lines) == 8
    assert len(val_lines) == 2
    assert train_lines[0] == "Line 0"
    assert val_lines[-1] == "Line 9"

def test_preprocessing_pipeline():
    """Verify preprocess_pipeline successfully reads, cleans, splits and writes."""
    train_path, val_path = preprocess_pipeline(RAW_FILE, PROCESSED_DIR, val_split=0.2)
    assert os.path.exists(train_path)
    assert os.path.exists(val_path)
    
    with open(train_path, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_path, 'r', encoding='utf-8') as f:
        val_lines = f.readlines()
        
    assert len(train_lines) > 0
    assert len(val_lines) > 0

def test_pytorch_dataset():
    """Verify that TextDataset loads preprocessed text correctly."""
    train_path = os.path.join(PROCESSED_DIR, 'train.txt')
    dataset = TextDataset(train_path)
    
    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, str)
    assert len(sample) > 0

def test_pytorch_dataloader():
    """Verify that get_dataloader produces batches of expected shape/type."""
    train_path = os.path.join(PROCESSED_DIR, 'train.txt')
    batch_size = 4
    dataloader = get_dataloader(train_path, batch_size=batch_size, shuffle=False)
    
    batch = next(iter(dataloader))
    
    assert isinstance(batch, list)
    assert len(batch) <= batch_size
    assert all(isinstance(s, str) for s in batch)
