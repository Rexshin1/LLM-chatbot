import math
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Projections for Query, Key, Value
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Attention and output dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        # Causal mask registration (lower triangular matrix)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, C] (Batch, SeqLen, d_model)
        B, T, C = x.size()
        
        # Project to Q, K, V and split into multiple heads
        # Shape: [B, T, C] -> [B, T, num_heads, head_dim] -> transpose -> [B, num_heads, T, head_dim]
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention: Q @ K^T / sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask: replace future tokens with -inf
        scores = scores.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        
        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Weighted sum of values: weights @ V
        out = torch.matmul(attn_weights, v)
        
        # Transpose and concatenate heads back to [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection and dropout
        out = self.out_proj(out)
        return self.out_dropout(out)
