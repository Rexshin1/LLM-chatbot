import os
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """
    A PyTorch Dataset that loads preprocessed text file and yields samples line-by-line.
    """
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Processed file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f.readlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> str:
        return self.lines[idx]

def get_dataloader(file_path: str, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """
    Returns a PyTorch DataLoader for the given preprocessed text file.
    """
    dataset = TextDataset(file_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader
