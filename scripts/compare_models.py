import os
import torch
from torch.utils.data import DataLoader
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer
from src.model import DecoderOnlyTransformer
from src.training import CausalLMDataset
from src.inference import Evaluator, TextGenerator

def evaluate_model(checkpoint_path, vocab_path, val_path):
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.load_vocab(vocab_path)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = DecoderOnlyTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    
    val_dataset = CausalLMDataset(val_path, tokenizer, seq_len=64)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    evaluator = Evaluator(model, device="cpu")
    loss, perp = evaluator.evaluate(val_loader)
    return loss, perp, model, tokenizer

def main():
    prompts = ["kecerdasan", "pytorch", "proses"]
    
    loss_v1, perp_v1, model_v1, tokenizer_v1 = evaluate_model(
        "checkpoints/best_model.pt",
        "data/tokenizer/vocab.json",
        "data/processed/val.txt"
    )
    
    loss_v2, perp_v2, model_v2, tokenizer_v2 = evaluate_model(
        "checkpoints/best_model_v2.pt",
        "data/tokenizer/vocab_v2.json",
        "data/processed/val_v2.txt"
    )
    
    generator_v1 = TextGenerator(model_v1, tokenizer_v1, device="cpu")
    generator_v2 = TextGenerator(model_v2, tokenizer_v2, device="cpu")
    
    print("\n=========================================")
    print("      MODEL COMPARISON: V1 VS V2         ")
    print("=========================================")
    print(f"BASELINE (v1):")
    print(f"  Validation Loss: {loss_v1:.4f}")
    print(f"  Perplexity:      {perp_v1:.4f}")
    print(f"IMPROVED (v2):")
    print(f"  Validation Loss: {loss_v2:.4f}")
    print(f"  Perplexity:      {perp_v2:.4f}")
    print("=========================================\n")
    
    print("--- Text Generation Comparison ---")
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        text_v1, _ = generator_v1.generate(prompt, max_new_tokens=25, temperature=0.7, top_k=5)
        text_v2, _ = generator_v2.generate(prompt, max_new_tokens=25, temperature=0.7, top_k=5)
        print(f"  BASELINE (v1): {text_v1}")
        print(f"  IMPROVED (v2): {text_v2}")
    print("----------------------------------\n")

if __name__ == "__main__":
    main()
