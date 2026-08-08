import os
import json
import torch
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer
from src.model import DecoderOnlyTransformer
from src.inference import TextGenerator

def main():
    checkpoint_path = "checkpoints/v3_best_model.pt"
    vocab_path = "data/tokenizer/vocab_v3.json"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
        
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.load_vocab(vocab_path)
    vocab_size = len(tokenizer.vocab)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = DecoderOnlyTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    
    generator = TextGenerator(model, tokenizer, device="cpu")
    
    prompts = ["kecerdasan", "pytorch", "proses", "machine learning", "jaringan", "data", "model"]
    temperatures = [0.7, 1.0, 1.2]
    top_ks = [5, 10, 20]
    max_tokens_list = [10, 20, 30]
    
    eos_id = tokenizer.token_to_id(tokenizer.eos_token)
    unk_id = tokenizer.token_to_id(tokenizer.unk_token)
    
    total_runs = 0
    total_new_tokens = 0
    stopped_by_eos = 0
    reached_max = 0
    total_unks = 0
    
    print("=========================================")
    print("      GENERATION QUALITY EVALUATION      ")
    print("=========================================")
    
    for prompt in prompts:
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        
        for temp in temperatures:
            for k in top_ks:
                for max_new in max_tokens_list:
                    text, token_ids = generator.generate(
                        prompt=prompt,
                        max_new_tokens=max_new,
                        temperature=temp,
                        top_k=k,
                        add_special_tokens=True
                    )
                    
                    # Extract only new generated tokens (after BOS + prompt tokens)
                    new_ids = token_ids[1 + prompt_len:]
                    new_tokens_count = len(new_ids)
                    
                    ended_with_eos = len(new_ids) > 0 and new_ids[-1] == eos_id
                    
                    if ended_with_eos:
                        stopped_by_eos += 1
                    else:
                        reached_max += 1
                        
                    num_unks = new_ids.count(unk_id)
                    total_unks += num_unks
                    
                    total_new_tokens += new_tokens_count
                    total_runs += 1
                    
                    # Print a clean subset of runs to keep logs readable (e.g. temp=0.7, top_k=5, max_new=20)
                    if temp == 0.7 and k == 5 and max_new == 20:
                        print(f"Prompt:        '{prompt}'")
                        print(f"Temp / Top-k:   {temp} / {k}")
                        print(f"Max New:        {max_new}")
                        print(f"Generated text: '{text}'")
                        print("-" * 40)
                        
    avg_len = total_new_tokens / total_runs if total_runs > 0 else 0.0
    eos_rate = stopped_by_eos / total_runs if total_runs > 0 else 0.0
    unk_rate = total_unks / total_new_tokens if total_new_tokens > 0 else 0.0
    
    print("\n=========================================")
    print("          GENERATION STATISTICS          ")
    print("=========================================")
    print(f"Total Runs:                 {total_runs}")
    print(f"Total New Tokens Generated: {total_new_tokens}")
    print(f"Average Generation Length:  {avg_len:.2f} tokens")
    print(f"Stopped by EOS:             {stopped_by_eos} ({eos_rate:.2%})")
    print(f"Reached Max Tokens:         {reached_max} ({reached_max/total_runs:.2%})")
    print(f"Total UNK Tokens Generated: {total_unks} ({unk_rate:.2%})")
    print("=========================================\n")

if __name__ == "__main__":
    main()
