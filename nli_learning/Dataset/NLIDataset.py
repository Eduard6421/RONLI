import json

from torch.utils.data import Dataset
from torch.utils.data import random_split



class NLIDataset(Dataset):

    def __init__(self, data_path, batch_size = 8, transform = None):
        self.data = self.load_data(data_path)
        self.batch_size = batch_size
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        guid = item["guid"]
        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]
        if(self.transform):
            sentence1 = self.transform(sentence1)
            sentence2 = self.transform(sentence2)
        label = item["label"]
        return guid, sentence1,sentence2, label
    
    def load_data(self, data_path):
        with open(data_path, "r",encoding="utf-8") as f:
            data = json.load(f)
        return data