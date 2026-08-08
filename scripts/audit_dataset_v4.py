import os
import json
import collections
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer

def main():
    raw_path = "data/raw/corpus_v4.txt"
    train_path = "data/processed/train_v4.txt"
    val_path = "data/processed/val_v4.txt"
    vocab_path = "data/tokenizer/vocab_v4.json"
    
    if not os.path.exists(raw_path):
        print(f"Error: Raw corpus file not found at {raw_path}")
        return
        
    with open(raw_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        
    raw_lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    
    num_chars = len(raw_text)
    num_words = len(raw_text.split())
    num_lines = len(raw_lines)
    
    # Read train and val
    with open(train_path, 'r', encoding='utf-8') as f:
        train_lines = [line.strip() for line in f.readlines() if line.strip()]
    with open(val_path, 'r', encoding='utf-8') as f:
        val_lines = [line.strip() for line in f.readlines() if line.strip()]
        
    # Overlap / Leakage
    train_set = set(train_lines)
    val_set = set(val_lines)
    overlap = train_set.intersection(val_set)
    
    # Duplicates check
    counter = collections.Counter(raw_lines)
    duplicates = [item for item, count in counter.items() if count > 1]
    duplicate_rate = len(duplicates) / len(raw_lines) if len(raw_lines) > 0 else 0.0
    
    # Tokenizer audit
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.load_vocab(vocab_path)
    vocab_size = len(tokenizer.vocab)
    
    # Tokenize whole corpus to get frequency
    all_tokens = []
    for line in raw_lines:
        all_tokens.extend(tokenizer._tokenize(line))
        
    token_freqs = collections.Counter(all_tokens)
    
    # UNK rate in validation set
    val_token_ids = []
    for line in val_lines:
        val_token_ids.extend(tokenizer.encode(line, add_special_tokens=False))
        
    unk_id = tokenizer.token_to_id(tokenizer.unk_token)
    num_unks = val_token_ids.count(unk_id)
    unk_rate = num_unks / len(val_token_ids) if len(val_token_ids) > 0 else 0.0
    
    print("=========================================")
    print("      DATASET V4 AUDIT SUMMARY           ")
    print("=========================================")
    print(f"Lines:                   {num_lines}")
    print(f"Characters:              {num_chars:,}")
    print(f"Words:                   {num_words:,}")
    print(f"Tokens:                  {len(all_tokens):,}")
    print(f"Vocabulary Size:         {vocab_size} tokens")
    print(f"UNK Rate (Val):          {unk_rate:.4%}")
    print(f"Duplicate Rate (Raw):    {duplicate_rate:.4%} ({len(duplicates)} duplicates)")
    print(f"Train/Val Overlap:       {len(overlap)} lines")
    
    # Token frequency distribution
    print("\nToken Frequency Distribution (Top 10):")
    for tok, freq in token_freqs.most_common(10):
        print(f"  '{tok}': {freq} times ({freq/len(all_tokens):.2%})")
    
    print("\nToken Frequency Distribution (Bottom 10):")
    for tok, freq in token_freqs.most_common()[-10:]:
        print(f"  '{tok}': {freq} times ({freq/len(all_tokens):.2%})")
    print("=========================================")

if __name__ == "__main__":
    main()
