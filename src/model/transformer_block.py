import torch
import torch.nn as nn
from src.model.attention import CausalSelfAttention
from src.model.feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, max_seq_len, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN structure for better training stability
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
