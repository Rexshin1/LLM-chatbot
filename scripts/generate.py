import os
import argparse
import torch
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer
from src.model import DecoderOnlyTransformer
from src.inference import TextGenerator

def main():
    parser = argparse.ArgumentParser(description="Autoregressive text generation from trained LLM checkpoint.")
    parser.add_argument("--prompt", type=str, required=True, help="Initial prompt for text generation.")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (lower = more deterministic).")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K sampling constraint (0 to disable).")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint file.")
    parser.add_argument("--vocab", type=str, default="data/tokenizer/vocab.json", help="Path to vocab JSON file.")
    args = parser.parse_args()
    
    vocab_path = args.vocab
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}.")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint}.")
        
    # Load tokenizer
    tokenizer = SimpleWordTokenizer(lowercase=True)
    tokenizer.load_vocab(vocab_path)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = checkpoint["model_config"]
    
    # Initialize model
    model = DecoderOnlyTransformer(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Initialize generator
    generator = TextGenerator(model, tokenizer, device="cpu")
    
    print(f"Loading model from {args.checkpoint}...")
    print(f"Prompt: {args.prompt}")
    
    # Generate text
    generated_text, generated_ids = generator.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        add_special_tokens=True
    )
    
    print("\n--- Generated Output ---")
    print(generated_text)
    print("------------------------")
    print(f"Token IDs: {generated_ids}")

if __name__ == "__main__":
    main()
