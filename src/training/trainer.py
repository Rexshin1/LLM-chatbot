import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

class Trainer:
    """
    Trainer class to manage training and validation loops, compute perplexity,
    clip gradients, and handle model checkpoints.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        grad_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        model_config: dict = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        self.model_config = model_config or {}
        
        self.best_val_loss = float("inf")
        self.start_epoch = 0
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            logits, loss = self.model(x, labels=y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
            # Optimizer step
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
        return avg_loss

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, loss = self.model(x, labels=y)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        return avg_loss

    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, filename: str = "best_model.pt"):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "model_config": self.model_config
        }
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)

    def load_checkpoint(self, filepath: str) -> dict:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
            
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["validation_loss"]
        
        return checkpoint

    def train(self, epochs: int):
        import time
        for epoch in range(self.start_epoch, epochs):
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            elapsed_time = time.time() - start_time
            
            # Compute perplexity
            perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Perplexity: {perplexity:.4f} | Time: {elapsed_time:.2f}s")
            
            # Save the best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, train_loss, val_loss, filename="best_model.pt")
                
            # Always save a latest checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss, filename="latest_model.pt")
            
            self.start_epoch = epoch + 1
