import os
import json
import torch
from torch.utils.data import DataLoader
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer
from src.model import DecoderOnlyTransformer
from src.training import Trainer, CausalLMDataset, set_seed

def main():
    set_seed(42)
    
    # Hyperparameters
    batch_size = 8
    seq_len = 64
    learning_rate = 0.001
    weight_decay = 0.01
    epochs = 15
    grad_clip = 1.0
    checkpoint_dir = "checkpoints"
    
    # Paths for v3
    vocab_path = "data/tokenizer/vocab_v3.json"
    train_path = "data/processed/train_v3.txt"
    val_path = "data/processed/val_v3.txt"
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found at {vocab_path}. Run build_data_v3 first.")
        
    # Load tokenizer
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.load_vocab(vocab_path)
    vocab_size = len(tokenizer.vocab)
    
    # Build Datasets & DataLoaders
    train_dataset = CausalLMDataset(train_path, tokenizer, seq_len=seq_len)
    val_dataset = CausalLMDataset(val_path, tokenizer, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize Model
    model_config = {
        "vocab_size": vocab_size,
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 4,
        "max_seq_len": 128,
        "dropout": 0.1
    }
    
    model = DecoderOnlyTransformer(**model_config)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Trainer
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
    
    # Override save_checkpoint to save with v3 suffix
    original_save = trainer.save_checkpoint
    def save_checkpoint_v3(epoch: int, train_loss: float, val_loss: float, filename: str):
        if filename == "best_model.pt":
            filename = "v3_best_model.pt"
        elif filename == "latest_model.pt":
            filename = "v3_latest_model.pt"
        original_save(epoch, train_loss, val_loss, filename=filename)
        
    trainer.save_checkpoint = save_checkpoint_v3
    
    # Resume v3 checkpoint if available
    latest_checkpoint_path = os.path.join(checkpoint_dir, "v3_latest_model.pt")
    if os.path.exists(latest_checkpoint_path):
        print(f"Found existing checkpoint at {latest_checkpoint_path}. Resuming training...")
        trainer.load_checkpoint(latest_checkpoint_path)
        
    print("Starting training on v3 dataset...")
    trainer.train(epochs=epochs)
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
