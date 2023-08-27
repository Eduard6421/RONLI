import json

from torch.utils.data import Dataset
from torch.utils.data import random_split


class NLIDatasetOversample(Dataset):

    def __init__(self, train_data, batch_size = 8, transform = None):
        self.data = train_data
        self.batch_size = batch_size
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def load_data(self, data_path):
        with open(data_path, "r",encoding="utf-8") as f:
            data = json.load(f)
        return data