import os
import json
import torch
from torch.utils.data import DataLoader
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer
from src.model import DecoderOnlyTransformer
from src.training import Trainer, CausalLMDataset, set_seed

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Hyperparameters
    batch_size = 8
    seq_len = 64
    learning_rate = 3e-4
    weight_decay = 0.01
    epochs = 10
    grad_clip = 1.0
    checkpoint_dir = "checkpoints"
    
    # Paths
    vocab_path = "data/tokenizer/vocab.json"
    train_path = "data/processed/train.txt"
    val_path = "data/processed/val.txt"
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found at {vocab_path}. Please run build_vocab first.")
        
    # Load tokenizer
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.load_vocab(vocab_path)
    vocab_size = len(tokenizer.vocab)
    
    # Build Datasets & DataLoaders
    train_dataset = CausalLMDataset(train_path, tokenizer, seq_len=seq_len)
    val_dataset = CausalLMDataset(val_path, tokenizer, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize Model & Optimizer
    model_config = {
        "vocab_size": vocab_size,
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 4,
        "max_seq_len": 128,  # must be >= seq_len (64)
        "dropout": 0.1
    }
    
    model = DecoderOnlyTransformer(**model_config)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        grad_clip=grad_clip,
        checkpoint_dir=checkpoint_dir,
        model_config=model_config
    )
    
    # Resume from latest checkpoint if available
    latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_model.pt")
    if os.path.exists(latest_checkpoint_path):
        print(f"Found existing checkpoint at {latest_checkpoint_path}. Resuming training...")
        trainer.load_checkpoint(latest_checkpoint_path)
        
    print("Starting training...")
    trainer.train(epochs=epochs)
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
