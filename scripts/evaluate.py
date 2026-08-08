import os
import torch
from torch.utils.data import DataLoader
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer
from src.model import DecoderOnlyTransformer
from src.training import CausalLMDataset
from src.inference import Evaluator

def main():
    checkpoint_path = "checkpoints/best_model.pt"
    vocab_path = "data/tokenizer/vocab.json"
    val_path = "data/processed/val.txt"
    seq_len = 64
    batch_size = 8
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}.")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}.")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation dataset not found at {val_path}.")
        
    # Load tokenizer
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.load_vocab(vocab_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    
    # Initialize model
    model = DecoderOnlyTransformer(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load validation dataset and dataloader
    val_dataset = CausalLMDataset(val_path, tokenizer, seq_len=seq_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Evaluate
    evaluator = Evaluator(model, device="cpu")
    loss, perplexity = evaluator.evaluate(val_loader)
    
    print("=========================================")
    print("        VALIDATION EVALUATION            ")
    print("=========================================")
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"Validation Loss: {loss:.4f}")
    print(f"Perplexity:      {perplexity:.4f}")
    print("=========================================")

if __name__ == "__main__":
    main()
