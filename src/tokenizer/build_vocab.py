import os
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer

def main():
    train_path = "data/processed/train.txt"
    val_path = "data/processed/val.txt"
    vocab_path = "data/tokenizer/vocab.json"
    
    # Read both train and validation datasets to build a complete vocab
    texts = []
    for path in [train_path, val_path]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
                
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.build_vocab(texts)
    tokenizer.save_vocab(vocab_path)
    
    print(f"Vocabulary successfully built and saved to {vocab_path}")
    print(f"Vocabulary size: {len(tokenizer.vocab)} tokens")

if __name__ == "__main__":
    main()
