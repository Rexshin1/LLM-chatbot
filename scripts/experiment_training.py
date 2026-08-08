import os
import json
import time
import math
import shutil
import torch
from torch.utils.data import DataLoader
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer
from src.model import DecoderOnlyTransformer
from src.training import Trainer, CausalLMDataset, set_seed

def run_experiment(lr: float, epochs: int, batch_size: int, exp_id: str) -> dict:
    set_seed(42)
    
    # Paths for v2
    vocab_path = "data/tokenizer/vocab_v2.json"
    train_path = "data/processed/train_v2.txt"
    val_path = "data/processed/val_v2.txt"
    checkpoint_dir = "checkpoints"
    
    # Load tokenizer
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.load_vocab(vocab_path)
    vocab_size = len(tokenizer.vocab)
    
    # Load dataset
    train_dataset = CausalLMDataset(train_path, tokenizer, seq_len=64)
    val_dataset = CausalLMDataset(val_path, tokenizer, seq_len=64)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model_config = {
        "vocab_size": vocab_size,
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 4,
        "max_seq_len": 128,
        "dropout": 0.1
    }
    model = DecoderOnlyTransformer(**model_config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
        grad_clip=1.0,
        checkpoint_dir=checkpoint_dir,
        model_config=model_config
    )
    
    checkpoint_filename = f"model_exp_{exp_id}.pt"
    
    # Override save_checkpoint to save with experimental id
    original_save = trainer.save_checkpoint
    def save_custom(epoch, train_loss, val_loss, filename):
        if filename == "best_model.pt":
            original_save(epoch, train_loss, val_loss, filename=checkpoint_filename)
        elif filename == "latest_model.pt":
            pass
            
    trainer.save_checkpoint = save_custom
    
    # Train
    start_time = time.time()
    trainer.train(epochs=epochs)
    duration = time.time() - start_time
    
    # Load best checkpoint to get final metrics
    best_path = os.path.join(checkpoint_dir, checkpoint_filename)
    checkpoint = torch.load(best_path, map_location="cpu")
    
    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint["validation_loss"]
    perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")
    
    result = {
        "exp_id": exp_id,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "perplexity": perplexity,
        "duration": duration,
        "checkpoint": best_path
    }
    return result

def main():
    experiments = [
        # Exp 1: Baseline config (LR 3e-4, 15 epochs, Batch 8)
        {"lr": 3e-4, "epochs": 15, "batch_size": 8, "exp_id": "1_baseline"},
        # Exp 2: Higher learning rate (LR 1e-3, 15 epochs, Batch 8)
        {"lr": 1e-3, "epochs": 15, "batch_size": 8, "exp_id": "2_higher_lr"},
        # Exp 3: Lower learning rate, longer training (LR 1e-4, 20 epochs, Batch 8)
        {"lr": 1e-4, "epochs": 20, "batch_size": 8, "exp_id": "3_lower_lr_longer"},
        # Exp 4: Larger batch size, moderate LR (LR 5e-4, 15 epochs, Batch 16)
        {"lr": 5e-4, "epochs": 15, "batch_size": 16, "exp_id": "4_large_batch"}
    ]
    
    results = []
    for exp in experiments:
        print(f"\nRunning Experiment {exp['exp_id']} (LR={exp['lr']}, Epochs={exp['epochs']}, BatchSize={exp['batch_size']})...")
        res = run_experiment(exp["lr"], exp["epochs"], exp["batch_size"], exp["exp_id"])
        results.append(res)
        print(f"Result -> Val Loss: {res['val_loss']:.4f} | Perplexity: {res['perplexity']:.4f} | Time: {res['duration']:.2f}s")
        
    # Sort results to find best validation loss
    results.sort(key=lambda x: x["val_loss"])
    best_exp = results[0]
    
    # Save a copy of the best experiment
    best_checkpoint_path = best_exp["checkpoint"]
    shutil_target = "checkpoints/best_experiment.pt"
    shutil.copy(best_checkpoint_path, shutil_target)
    
    print("\n=========================================")
    print("      EXPERIMENT SUMMARY RESULTS         ")
    print("=========================================")
    for r in results:
        print(f"Exp {r['exp_id']}: LR={r['lr']}, Epochs={r['epochs']}, Batch={r['batch_size']} -> Val Loss: {r['val_loss']:.4f} | Perp: {r['perplexity']:.4f} | Time: {r['duration']:.2f}s")
    print("=========================================")
    print(f"BEST CONFIGURATION: Exp {best_exp['exp_id']}")
    print(f"Best Val Loss:      {best_exp['val_loss']:.4f}")
    print(f"Best Perplexity:    {best_exp['perplexity']:.4f}")
    print(f"Checkpoint saved:   {shutil_target}")
    print("=========================================\n")

if __name__ == "__main__":
    main()
