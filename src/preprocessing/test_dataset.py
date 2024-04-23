import torch
from torch.utils.data import Dataset
import pandas as pd

class TestDataset(Dataset):
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data.iloc[idx][["x1", "x2"]].values
        input = torch.tensor(input, dtype=torch.float)
        target = self.data.iloc[idx]["y"]
        target = torch.tensor(target, dtype=torch.float)
        return input, target