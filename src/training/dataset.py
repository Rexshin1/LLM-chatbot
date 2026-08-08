import torch
from torch.utils.data import Dataset
from src.tokenizer.simple_tokenizer import SimpleWordTokenizer

class CausalLMDataset(Dataset):
    """
    A PyTorch Dataset that loads a text file, tokenizes it, and constructs
    input-target sequence pairs for causal language modeling.
    """
    def __init__(self, file_path: str, tokenizer: SimpleWordTokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # Read lines from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        # Tokenize and concatenate all lines with BOS and EOS tokens
        all_ids = []
        for line in lines:
            # We encode each line individually adding BOS/EOS
            ids = tokenizer.encode(line, add_special_tokens=True)
            all_ids.extend(ids)
            
        self.inputs = []
        self.targets = []
        
        pad_id = tokenizer.token_to_id(tokenizer.pad_token)
        
        # If the whole corpus is smaller than one sequence length + 1
        if len(all_ids) < seq_len + 1:
            chunk = all_ids + [pad_id] * (seq_len + 1 - len(all_ids))
            self.inputs.append(chunk[:-1])
            self.targets.append(chunk[1:])
        else:
            # Chunk the corpus into non-overlapping seq_len segments
            for i in range(0, len(all_ids) - seq_len, seq_len):
                chunk = all_ids[i:i + seq_len + 1]
                if len(chunk) < seq_len + 1:
                    chunk = chunk + [pad_id] * (seq_len + 1 - len(chunk))
                self.inputs.append(chunk[:-1])
                self.targets.append(chunk[1:])

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.inputs[idx], dtype=torch.long)
        y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y
