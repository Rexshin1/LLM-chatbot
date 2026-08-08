import torch
import torch.nn as nn
from src.model.embeddings import TransformerEmbeddings
from src.model.transformer_block import TransformerBlock

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, num_heads: int = 4, 
                 num_layers: int = 4, max_seq_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # Embeddings (Token + Positional)
        self.embeddings = TransformerEmbeddings(vocab_size, d_model, max_seq_len, dropout)
        
        # Transformer Blocks Stack
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, max_seq_len, dropout)
            for _ in range(num_layers)
        ])
        
        # Final LayerNorm
        self.ln_f = nn.LayerNorm(d_model)
        
        # LM Head (projects hidden states back to vocabulary logit space)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # input_ids shape: [batch_size, seq_len]
        batch_size, seq_len = input_ids.size()
        
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length ({seq_len}) exceeds max sequence length ({self.max_seq_len})")
            
        # Get embeddings
        x = self.embeddings(input_ids) # [batch_size, seq_len, d_model]
        
        # Pass through blocks
        for block in self.blocks:
            x = block(x)
            
        # Final LayerNorm
        x = self.ln_f(x) # [batch_size, seq_len, d_model]
        
        # Project to vocabulary size
        logits = self.lm_head(x) # [batch_size, seq_len, vocab_size]
        
        loss = None
        if labels is not None:
            # Shift inputs and labels for next-token prediction
            # logits shape: [batch_size, seq_len - 1, vocab_size]
            # labels shape: [batch_size, seq_len - 1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten tensors
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return logits, loss
