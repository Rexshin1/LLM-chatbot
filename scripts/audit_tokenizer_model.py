import os
import json
import torch
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer
from src.model import DecoderOnlyTransformer

def main():
    vocab_v3_path = "data/tokenizer/vocab_v3.json"
    vocab_v2_path = "data/tokenizer/vocab_v2.json"
    vocab_v1_path = "data/tokenizer/vocab.json"
    checkpoint_path = "checkpoints/v3_best_model.pt"
    
    print("=========================================")
    print("      TOKENIZER & MODEL AUDIT            ")
    print("=========================================")
    
    # 1. Vocab v3 properties
    if os.path.exists(vocab_v3_path):
        with open(vocab_v3_path, 'r', encoding='utf-8') as f:
            vocab_v3 = json.load(f)
        print(f"Vocab V3 size:           {len(vocab_v3)} tokens")
        ids = list(vocab_v3.values())
        print(f"Vocab V3 min ID:         {min(ids)}")
        print(f"Vocab V3 max ID:         {max(ids)}")
        print("Special tokens in V3:")
        for token in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            print(f"  {token}: {vocab_v3.get(token)}")
    else:
        print("Vocab V3 file not found!")
        vocab_v3 = {}

    # 2. Checkpoint v3 properties
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = checkpoint["model_config"]
        print(f"\nCheckpoint V3 config:")
        print(f"  vocab_size:            {model_config.get('vocab_size')}")
        print(f"  d_model:               {model_config.get('d_model')}")
        print(f"  num_layers:            {model_config.get('num_layers')}")
        
        # Embeddings & LM head dimensions in state_dict
        state_dict = checkpoint["model_state_dict"]
        tok_embed_weight = state_dict["embeddings.tok_embed.weight"]
        lm_head_weight = state_dict["lm_head.weight"]
        print(f"State dict shapes:")
        print(f"  Token Embedding shape: {list(tok_embed_weight.shape)}")
        print(f"  LM Head weight shape:  {list(lm_head_weight.shape)}")
        
        # Check matching
        mismatch = model_config.get('vocab_size') != len(vocab_v3)
        print(f"\nVocab V3 size matches model config: {'YES' if not mismatch else 'NO'}")
        
        embed_mismatch = tok_embed_weight.shape[0] != len(vocab_v3)
        print(f"Embedding size matches Vocab V3:    {'YES' if not embed_mismatch else 'NO'}")
    else:
        print("Checkpoint V3 file not found!")
        
    # 3. Check for any token ID out of range
    if vocab_v3:
        out_of_range = [tok for tok, idx in vocab_v3.items() if idx < 0 or idx >= len(vocab_v3)]
        print(f"Token IDs out of range [0, {len(vocab_v3)-1}]: {len(out_of_range)}")
        
    # 4. Compare with Vocab V1/V2
    if os.path.exists(vocab_v1_path):
        with open(vocab_v1_path, 'r', encoding='utf-8') as f:
            vocab_v1 = json.load(f)
        print(f"\nVocab V1 (baseline) size: {len(vocab_v1)} tokens")
        print(f"  'yang' ID in V1:        {vocab_v1.get('yang')}")
    if os.path.exists(vocab_v3_path):
        print(f"  'yang' ID in V3:        {vocab_v3.get('yang')}")

    print("=========================================\n")

if __name__ == "__main__":
    main()
