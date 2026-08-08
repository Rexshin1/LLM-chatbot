import os
import re

def clean_text(text: str) -> str:
    """
    Cleans text by removing excessive whitespace and basic character cleaning.
    """
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove lines that are purely whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

def normalize_text(text: str, lowercase: bool = True) -> str:
    """
    Normalizes text by lowercasing (optional) and removing uncommon special characters.
    """
    if lowercase:
        text = text.lower()
    return text

def split_data(text: str, val_split: float = 0.2) -> tuple[str, str]:
    """
    Splits the text into training and validation sets.
    Splits are done line-by-line to preserve line/sentence integrity.
    """
    lines = text.split('\n')
    split_idx = int(len(lines) * (1 - val_split))
    
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    return '\n'.join(train_lines), '\n'.join(val_lines)

def preprocess_pipeline(raw_file: str, processed_dir: str, val_split: float = 0.2, lowercase: bool = True):
    """
    Full pipeline to read raw text, clean, normalize, split, and save processed datasets.
    """
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Raw file not found: {raw_file}")
        
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Read dataset
    with open(raw_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        
    # 2. Basic cleaning
    cleaned = clean_text(raw_text)
    
    # 3. Normalization
    normalized = normalize_text(cleaned, lowercase=lowercase)
    
    # 4. Split
    train_data, val_data = split_data(normalized, val_split=val_split)
    
    # 5. Save results
    train_path = os.path.join(processed_dir, 'train.txt')
    val_path = os.path.join(processed_dir, 'val.txt')
    
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write(train_data)
        
    with open(val_path, 'w', encoding='utf-8') as f:
        f.write(val_data)
        
    print(f"Preprocessing completed successfully!")
    print(f"Train samples: {len(train_data.split('\n'))} lines saved to {train_path}")
    print(f"Validation samples: {len(val_data.split('\n'))} lines saved to {val_path}")
    
    return train_path, val_path

if __name__ == "__main__":
    # Run the preprocess pipeline directly if script is executed
    raw_path = "data/raw/corpus.txt"
    processed_directory = "data/processed"
    preprocess_pipeline(raw_path, processed_directory)
