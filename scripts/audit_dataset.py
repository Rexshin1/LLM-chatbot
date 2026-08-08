import os
import argparse
import json
import collections
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer

def main():
    parser = argparse.ArgumentParser(description="Audit dataset and tokenizer vocabulary.")
    parser.add_argument("--version", type=str, default="v2", choices=["v2", "v3"], help="Dataset version to audit.")
    args = parser.parse_args()
    
    suffix = f"_{args.version}"
    raw_path = f"data/raw/corpus{suffix}.txt"
    train_path = f"data/processed/train{suffix}.txt"
    val_path = f"data/processed/val{suffix}.txt"
    vocab_path = f"data/tokenizer/vocab{suffix}.json"
    
    if not os.path.exists(raw_path):
        print(f"Error: Raw corpus file not found at {raw_path}")
        return
        
    # Read raw corpus
    with open(raw_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        
    raw_lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    
    # Character, word, and line count
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
    
    # Duplicates in raw corpus
    duplicates = [item for item, count in collections.Counter(raw_lines).items() if count > 1]
    
    # Tokenizer audit
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.load_vocab(vocab_path)
    vocab_size = len(tokenizer.vocab)
    
    # Tokenize whole corpus to get frequency
    all_tokens = []
    for line in raw_lines:
        all_tokens.extend(tokenizer._tokenize(line))
        
    token_freqs = collections.Counter(all_tokens)
    rare_tokens = [tok for tok, freq in token_freqs.items() if freq == 1]
    
    # UNK proportion in validation set
    val_token_ids = []
    for line in val_lines:
        val_token_ids.extend(tokenizer.encode(line, add_special_tokens=False))
        
    unk_id = tokenizer.token_to_id(tokenizer.unk_token)
    num_unks = val_token_ids.count(unk_id)
    unk_rate = num_unks / len(val_token_ids) if len(val_token_ids) > 0 else 0.0
    
    # Line length stats (by tokens)
    lengths = [len(tokenizer._tokenize(line)) for line in raw_lines]
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    
    print(f"=========================================")
    print(f"   DATASET & TOKENIZER AUDIT ({args.version.upper()})    ")
    print(f"=========================================")
    print(f"1. Raw Lines:            {num_lines}")
    print(f"2. Total Characters:     {num_chars:,}")
    print(f"3. Total Words:          {num_words:,}")
    print(f"4. Total Tokens (raw):   {len(all_tokens):,}")
    print(f"5. Vocabulary Size:      {vocab_size} tokens")
    print(f"6. Duplicate Lines (raw):{len(duplicates)}")
    print(f"7. Train/Val Overlap:    {len(overlap)} lines")
    print(f"8. Line Lengths (tokens): Min={min_len}, Max={max_len}, Avg={avg_len:.2f}")
    print(f"9. Special Tokens count: {len(tokenizer.special_tokens)}")
    print(f"10. Validation UNK Rate: {unk_rate:.4%} ({num_unks} / {len(val_token_ids)} tokens)")
    print(f"11. Rare Tokens (freq=1):{len(rare_tokens)} ({len(rare_tokens)/vocab_size:.2%})")
    print("=========================================")
    
    # Problems checklist
    print("\n--- Identified Issues ---")
    issues = []
    if len(duplicates) > 0:
        issues.append(f"Duplicate text lines present in raw corpus ({len(duplicates)} lines).")
    if len(overlap) > 0:
        issues.append(f"Data leakage! {len(overlap)} lines exist in both train and validation splits.")
    if unk_rate > 0.15: # Raised for v3 since we filtered rare tokens
        issues.append(f"High validation UNK rate ({unk_rate:.2%}). Tokenizer did not cover validation words well.")
    if len(rare_tokens) / vocab_size > 0.5:
        issues.append(f"More than 50% of vocabulary consists of single-occurrence rare tokens ({len(rare_tokens)} / {vocab_size}).")
        
    if not issues:
        print("No significant dataset or tokenizer issues found!")
    else:
        for issue in issues:
            print(f"- [WARNING] {issue}")
    print("-------------------------\n")

if __name__ == "__main__":
    main()
