import torch
import torch.nn as nn
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer

class TextGenerator:
    """
    TextGenerator class to perform autoregressive decoding from a prompt
    using temperature scaling and top-k filtering on logits.
    """
    def __init__(self, model: nn.Module, tokenizer: SimpleWordTokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        add_special_tokens: bool = True
    ) -> tuple[str, list[int]]:
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        eos_id = self.tokenizer.token_to_id(self.tokenizer.eos_token)
        if len(input_ids) > 0 and input_ids[-1] == eos_id:
            input_ids = input_ids[:-1]
            
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device) # [1, seq_len]
        generated = list(input_ids)
        
        for _ in range(max_new_tokens):
            # Crop inputs if they exceed maximum sequence length of the model
            curr_tensor = input_tensor[:, -self.model.max_seq_len:]
            
            with torch.no_grad():
                logits, _ = self.model(curr_tensor)
                
            # Extract last token logits: [1, seq_len, vocab_size] -> [vocab_size]
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                
            # Apply Top-K filtering
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[-1]] = float("-inf")
                
            # Convert logits to probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            if temperature == 0.0 or top_k == 1:
                next_token_id = torch.argmax(probs).item()
            else:
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
            generated.append(next_token_id)
            
            if next_token_id == eos_id:
                break
                
            # Append next_token_id to input_tensor for next iteration
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
            input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)
            
        new_tokens = generated[len(input_ids):]
        decoded_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return decoded_text, generated
