import os
import torch
from src.tokenizer.simple_tokenizer import SimpleWordTokenizerV5
from src.model import DecoderOnlyTransformer
from src.inference.generator import TextGenerator

def main():
    checkpoint_path = "checkpoints/rexa_v5_instruction.pt"
    vocab_path = "data/tokenizer/vocab_v5.json"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}. Train the model first.")
        return
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file not found at {vocab_path}.")
        return
        
    print("Loading REXA V5 model...")
    tokenizer = SimpleWordTokenizerV5(lowercase=True)
    tokenizer.load_vocab(vocab_path)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = DecoderOnlyTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    generator = TextGenerator(model, tokenizer, device="cpu")
    print("Model loaded successfully.\n")
    print("=" * 60)
    print("REXA SYSTEM EVALUATION REPORT")
    print("=" * 60)
    
    eval_cases = [
        {"category": "GENERAL/CONVERSATION", "prompt": "<|system|>Kamu adalah REXA, asisten AI dari REXSHIN.<|user|>halo bro, lu siapa?<|assistant|>"},
        {"category": "INSTRUCTION", "prompt": "<|system|>Kamu adalah REXA, asisten AI dari REXSHIN.<|user|>tolong ubah kalimat ini jadi lebih profesional: gue mau resign besok<|assistant|>"},
        {"category": "TYPO", "prompt": "<|system|>Kamu adalah REXA, asisten AI dari REXSHIN.<|user|>ap itu transfrmer?<|assistant|>"},
        {"category": "EXPLANATION", "prompt": "<|system|>Kamu adalah REXA, asisten AI dari REXSHIN.<|user|>jelasin kenapa langit warnanya biru<|assistant|>"},
        {"category": "WRITING", "prompt": "<|system|>Kamu adalah REXA, asisten AI dari REXSHIN.<|user|>buatkan cerita pendek tentang programmer<|assistant|>"},
        {"category": "REASONING", "prompt": "<|system|>Kamu adalah REXA, asisten AI dari REXSHIN.<|user|>siapa presiden pertama indonesia?<|assistant|>"},
        {"category": "MATH", "prompt": "<|system|>Kamu adalah REXA, asisten AI dari REXSHIN.<|user|>15 dikali 3 berapa?<|assistant|>"},
        {"category": "CODING", "prompt": "<|system|>Kamu adalah REXA, asisten AI dari REXSHIN.<|user|>bikin landing page html sederhana<|assistant|>"},
        {
            "category": "MULTI-TURN",
            "prompt": "<|system|>Kamu adalah REXA, asisten AI dari REXSHIN.<|user|>apa itu ai?<|assistant|>ai adalah kecerdasan buatan.<|user|>terus machine learning?<|assistant|>"
        }
    ]
    
    for case in eval_cases:
        cat = case["category"]
        p = case["prompt"]
        print(f"\n[Category: {cat}]")
        # Extract user query from prompt for clean printing
        query = p.split("<|user|>")[-1].split("<|assistant|>")[0]
        if "<|user|>" in p:
            print(f"Prompt: {query}")
        else:
            print(f"Prompt: {p}")
            
        try:
            text, _ = generator.generate(
                prompt=p,
                max_new_tokens=40,
                temperature=0.3, # low temperature for more focused evaluation
                top_k=5,
                add_special_tokens=True
            )
            # Clean V5 special tags if outputted
            for tag in ["<|system|>", "<|user|>", "<|assistant|>", "<|eos|>", "<eos>"]:
                text = text.replace(tag, "")
                text = text.replace(tag.upper(), "")
            text = text.strip()
            print(f"Generated: {text}")
        except Exception as e:
            print(f"Error generating: {str(e)}")
        print("-" * 40)
        
if __name__ == "__main__":
    main()
