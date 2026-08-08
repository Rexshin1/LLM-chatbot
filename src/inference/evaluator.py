import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Evaluator:
    """
    Evaluator class to calculate evaluation loss and perplexity on a validation dataset.
    """
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        
    def evaluate(self, val_loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, loss = self.model(x, labels=y)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        
        return avg_loss, perplexity
