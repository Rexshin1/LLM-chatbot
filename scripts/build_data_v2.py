import os
from src.dataset.preprocess import preprocess_pipeline
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer

def main():
    raw_path = "data/raw/corpus_v2.txt"
    processed_dir = "data/processed"
    
    # 1. Preprocess
    print("Preprocessing corpus_v2.txt...")
    train_path, val_path = preprocess_pipeline(raw_path, processed_dir, val_split=0.2)
    
    # Rename output files to avoid overwriting baseline files
    new_train_path = os.path.join(processed_dir, "train_v2.txt")
    new_val_path = os.path.join(processed_dir, "val_v2.txt")
    
    if os.path.exists(train_path):
        os.replace(train_path, new_train_path)
    if os.path.exists(val_path):
        os.replace(val_path, new_val_path)
        
    print(f"Saved v2 datasets to {new_train_path} and {new_val_path}")
    
    # 2. Build Vocab
    vocab_path = "data/tokenizer/vocab_v2.json"
    print("Building vocabulary for v2 dataset...")
    
    texts = []
    for path in [new_train_path, new_val_path]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
                
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.build_vocab(texts)
    tokenizer.save_vocab(vocab_path)
    
    print(f"Saved v2 vocabulary to {vocab_path}")
    print(f"Vocabulary size: {len(tokenizer.vocab)} tokens")

if __name__ == "__main__":
    main()
