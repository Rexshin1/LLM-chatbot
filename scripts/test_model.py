import os
import json
import torch
from src.model import DecoderOnlyTransformer

def main():
    # Load vocabulary from Phase 3 to get actual vocab size
    vocab_path = "data/tokenizer/vocab.json"
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        vocab_size = len(vocab)
    else:
        print("[Warning] actual vocab file not found. Using default vocab_size=256.")
        vocab_size = 256
        
    # Model configuration as requested
    d_model = 128
    num_heads = 4
    num_layers = 4
    max_seq_len = 128
    dropout = 0.1
    
    # Initialize model
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        dropout=dropout
    )
    
    # Total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Device
    device = "cpu"
    
    # Prepare dummy input (batch_size=2, seq_len=10)
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    # Forward pass
    logits, loss = model(input_ids, labels=labels)
    
    print("=========================================")
    print("        TRANSFORMER MODEL AUDIT          ")
    print("=========================================")
    print(f"Device:           {device}")
    print(f"Vocab Size:       {vocab_size}")
    print(f"Total Params:     {total_params:,}")
    print(f"Trainable Params: {trainable_params:,}")
    print(f"Input Shape:      {list(input_ids.shape)} (batch_size, seq_len)")
    print(f"Output Shape:     {list(logits.shape)} (batch_size, seq_len, vocab_size)")
    if loss is not None:
        print(f"Loss (Causal LM): {loss.item():.4f}")
    else:
        print("Loss (Causal LM): N/A")
    print("=========================================")

if __name__ == "__main__":
    main()
