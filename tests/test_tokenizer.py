import os
import pytest
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer

VOCAB_TEST_PATH = "data/tokenizer/vocab_test.json"

def test_tokenizer_vocab_build():
    """Verify that vocabulary can be built from raw text."""
    texts = ["Halo dunia!", "Ini adalah tes tokenizer."]
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.build_vocab(texts)
    
    assert len(tokenizer.vocab) > 4
    for token in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
        assert token in tokenizer.vocab
        assert tokenizer.vocab[token] < 4

def test_tokenizer_encode_decode():
    """Verify that encode and decode work correctly and are invertible."""
    texts = ["Halo dunia!", "Ini adalah tes tokenizer."]
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.build_vocab(texts)
    
    original_text = "halo dunia!"
    encoded = tokenizer.encode(original_text, add_special_tokens=False)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    
    assert isinstance(encoded, list)
    assert all(isinstance(idx, int) for idx in encoded)
    assert decoded == original_text

def test_special_tokens():
    """Verify BOS and EOS tokens are added correctly when requested."""
    texts = ["Halo dunia!"]
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.build_vocab(texts)
    
    encoded = tokenizer.encode("Halo dunia!", add_special_tokens=True)
    assert encoded[0] == tokenizer.vocab["<BOS>"]
    assert encoded[-1] == tokenizer.vocab["<EOS>"]
    
    decoded_skip = tokenizer.decode(encoded, skip_special_tokens=True)
    assert decoded_skip == "halo dunia!"
    
    decoded_no_skip = tokenizer.decode(encoded, skip_special_tokens=False)
    assert "<BOS>" in decoded_no_skip
    assert "<EOS>" in decoded_no_skip

def test_unknown_token():
    """Verify that out-of-vocabulary words resolve to <UNK>."""
    texts = ["Halo dunia!"]
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.build_vocab(texts)
    
    encoded = tokenizer.encode("bukan halo", add_special_tokens=False)
    assert encoded[0] == tokenizer.vocab["<UNK>"]
    assert encoded[1] == tokenizer.vocab["halo"]

def test_save_load_vocab():
    """Verify that vocabulary can be saved to disk and reloaded correctly."""
    texts = ["Halo dunia!", "Tes saving."]
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.build_vocab(texts)
    
    tokenizer.save_vocab(VOCAB_TEST_PATH)
    assert os.path.exists(VOCAB_TEST_PATH)
    
    new_tokenizer = SimpleWordTokenizer(lowercase=True)
    new_tokenizer.load_vocab(VOCAB_TEST_PATH)
    
    assert new_tokenizer.vocab == tokenizer.vocab
    assert new_tokenizer.id_to_token == tokenizer.id_to_token
    
    if os.path.exists(VOCAB_TEST_PATH):
        os.remove(VOCAB_TEST_PATH)

def test_simple_verification_flow():
    """
    Verify the flow: teks -> token -> token ID -> decode -> teks kembali
    """
    vocab_path = "data/tokenizer/vocab.json"
    tokenizer = SimpleWordTokenizer(lowercase=True)
    
    assert os.path.exists(vocab_path), "Actual vocabulary file must exist."
    tokenizer.load_vocab(vocab_path)
    
    sample_text = "kecerdasan buatan atau artificial intelligence saat ini berkembang sangat pesat."
    
    # 1. Teks -> Token
    tokens = tokenizer._tokenize(sample_text)
    print(f"\n[Flow] Tokens: {tokens}")
    assert len(tokens) > 0
    
    # 2. Token -> Token ID
    encoded = tokenizer.encode(sample_text, add_special_tokens=False)
    print(f"[Flow] Token IDs: {encoded}")
    assert len(encoded) == len(tokens)
    
    # 3. Decode -> Teks kembali
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"[Flow] Decoded text: {decoded}")
    
    assert decoded == sample_text.lower()

def test_tokenizer_min_freq():
    """Verify that build_vocab correctly filters tokens based on min_freq."""
    texts = ["halo dunia", "halo semua", "semua belajar"]
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.build_vocab(texts, min_freq=2)
    
    assert "halo" in tokenizer.vocab
    assert "semua" in tokenizer.vocab
    assert "dunia" not in tokenizer.vocab
    assert "belajar" not in tokenizer.vocab
    
    encoded = tokenizer.encode("dunia", add_special_tokens=False)
    assert encoded[0] == tokenizer.vocab["<UNK>"]
