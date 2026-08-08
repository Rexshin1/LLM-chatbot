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
    
    loss_v2, perp_v2, model_v2, tokenizer_v2 = evaluate_model(
        "checkpoints/best_experiment.pt",
        "data/tokenizer/vocab_v2.json",
        "data/processed/val_v2.txt"
    )
    
    loss_v3, perp_v3, model_v3, tokenizer_v3 = evaluate_model(
        "checkpoints/v3_best_model.pt",
        "data/tokenizer/vocab_v3.json",
        "data/processed/val_v3.txt"
    )
    
    generator_v2 = TextGenerator(model_v2, tokenizer_v2, device="cpu")
    generator_v3 = TextGenerator(model_v3, tokenizer_v3, device="cpu")
    
    print("\n=========================================")
    print("      MODEL COMPARISON: V2 VS V3         ")
    print("=========================================")
    print(f"IMPROVED V2 (from Exp 2):")
    print(f"  Validation Loss: {loss_v2:.4f}")
    print(f"  Perplexity:      {perp_v2:.4f}")
    print(f"IMPROVED V3 (Vocab filtered):")
    print(f"  Validation Loss: {loss_v3:.4f}")
    print(f"  Perplexity:      {perp_v3:.4f}")
    print("=========================================\n")
    
    print("--- Text Generation Comparison ---")
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        text_v2, _ = generator_v2.generate(prompt, max_new_tokens=25, temperature=0.7, top_k=5)
        text_v3, _ = generator_v3.generate(prompt, max_new_tokens=25, temperature=0.7, top_k=5)
        print(f"  V2: {text_v2}")
        print(f"  V3: {text_v3}")
    print("----------------------------------\n")

if __name__ == "__main__":
    main()
