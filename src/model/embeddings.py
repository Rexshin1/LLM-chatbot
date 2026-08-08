import torch
import torch.nn as nn

class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len]
        seq_len = x.size(1)
        # Create positions: [1, seq_len]
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        
        token_emb = self.tok_embed(x) # [batch_size, seq_len, d_model]
        pos_emb = self.pos_embed(positions) # [1, seq_len, d_model]
        
        embeddings = token_emb + pos_emb
        return self.dropout(embeddings)
