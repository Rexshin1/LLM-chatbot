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
    prompts = ["kecerdasan", "pytorch", "proses", "machine learning", "jaringan", "data", "model", "transformer", "python"]
    
    loss_v3, perp_v3, model_v3, tokenizer_v3 = evaluate_model(
        "checkpoints/v3_best_model.pt",
        "data/tokenizer/vocab_v3.json",
        "data/processed/val_v3.txt"
    )
    
    loss_v4, perp_v4, model_v4, tokenizer_v4 = evaluate_model(
        "checkpoints/v4_best_model.pt",
        "data/tokenizer/vocab_v4.json",
        "data/processed/val_v4.txt"
    )
    
    generator_v3 = TextGenerator(model_v3, tokenizer_v3, device="cpu")
    generator_v4 = TextGenerator(model_v4, tokenizer_v4, device="cpu")
    
    print("\n=========================================")
    print("      MODEL COMPARISON: V3 VS V4         ")
    print("=========================================")
    print(f"IMPROVED V3 (Vocab filtered):")
    print(f"  Validation Loss: {loss_v3:.4f}")
    print(f"  Perplexity:      {perp_v3:.4f}")
    print(f"IMPROVED V4 (Corpus expanded):")
    print(f"  Validation Loss: {loss_v4:.4f}")
    print(f"  Perplexity:      {perp_v4:.4f}")
    print("=========================================\n")
    
    print("--- Text Generation Comparison ---")
    
    eos_id = tokenizer_v4.token_to_id(tokenizer_v4.eos_token)
    unk_id = tokenizer_v4.token_to_id(tokenizer_v4.unk_token)
    
    total_runs = 0
    total_new_tokens = 0
    stopped_by_eos = 0
    total_unks = 0
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        text_v3, _ = generator_v3.generate(prompt, max_new_tokens=30, temperature=0.7, top_k=10)
        text_v4, token_ids = generator_v4.generate(prompt, max_new_tokens=30, temperature=0.7, top_k=10, add_special_tokens=True)
        
        prompt_len = len(tokenizer_v4.encode(prompt, add_special_tokens=False))
        new_ids = token_ids[1 + prompt_len:]
        new_tokens_count = len(new_ids)
        
        ended_with_eos = len(new_ids) > 0 and new_ids[-1] == eos_id
        if ended_with_eos:
            stopped_by_eos += 1
            
        num_unks = new_ids.count(unk_id)
        total_unks += num_unks
        total_new_tokens += new_tokens_count
        total_runs += 1
        
        print(f"  V3: {text_v3}")
        print(f"  V4: {text_v4}")
        
    avg_len = total_new_tokens / total_runs if total_runs > 0 else 0.0
    eos_rate = stopped_by_eos / total_runs if total_runs > 0 else 0.0
    unk_rate = total_unks / total_new_tokens if total_new_tokens > 0 else 0.0
    
    print("\n=========================================")
    print("       V4 GENERATION STATISTICS          ")
    print("=========================================")
    print(f"Average Generation Length:  {avg_len:.2f} tokens")
    print(f"EOS Rate:                   {eos_rate:.2%}")
    print(f"UNK Rate:                   {unk_rate:.2%}")
    print("=========================================\n")

if __name__ == "__main__":
    main()
