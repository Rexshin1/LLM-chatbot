import os
import shutil
from src.dataset.preprocess import preprocess_pipeline
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer

def main():
    raw_path_v2 = "data/raw/corpus_v2.txt"
    raw_path_v3 = "data/raw/corpus_v3.txt"
    processed_dir = "data/processed"
    
    # 1. Copy raw file
    if not os.path.exists(raw_path_v3):
        shutil.copy(raw_path_v2, raw_path_v3)
        print(f"Copied {raw_path_v2} to {raw_path_v3}")
        
    # 2. Preprocess
    print("Preprocessing corpus_v3.txt...")
    train_path, val_path = preprocess_pipeline(raw_path_v3, processed_dir, val_split=0.2)
    
    # Rename output files to avoid overwriting baseline files
    new_train_path = os.path.join(processed_dir, "train_v3.txt")
    new_val_path = os.path.join(processed_dir, "val_v3.txt")
    
    if os.path.exists(train_path):
        os.replace(train_path, new_train_path)
    if os.path.exists(val_path):
        os.replace(val_path, new_val_path)
        
    print(f"Saved v3 datasets to {new_train_path} and {new_val_path}")
    
    # 3. Build Vocab with min_freq=2 to eliminate rare tokens
    vocab_path = "data/tokenizer/vocab_v3.json"
    print("Building vocabulary for v3 dataset (min_freq=2)...")
    
    texts = []
    for path in [new_train_path, new_val_path]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
                
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.build_vocab(texts, min_freq=2)
    tokenizer.save_vocab(vocab_path)
    
    print(f"Saved v3 vocabulary to {vocab_path}")
    print(f"Vocabulary size: {len(tokenizer.vocab)} tokens")

if __name__ == "__main__":
    main()
