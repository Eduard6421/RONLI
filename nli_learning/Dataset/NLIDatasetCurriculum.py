import json
import random

from torch.utils.data import Dataset
from torch.utils.data import random_split


class NLIDatasetCurriculum(Dataset):
    def __init__(self, data_path, batch_size=8, transform=None, train_stage=0):
        self.data = self.load_data(data_path)
        self.batch_size = batch_size
        self.transform = transform
        self.fractions = [0.3, 0.6, 1]
        self.unlocked_lengths = [
            int(len(self.data) * fraction) for fraction in self.fractions
        ]
        self.current_unlock_index = train_stage

        self.data_subset = self.data[: self.unlocked_lengths[self.current_unlock_index]]
        self.data_subset = self._oversample(self.data_subset)

    def _oversample(self, data):
        # Precomputed oversampling

        type_of_each = {}

        maxim = 0
        for item in data:
            label = item["label"]
            if label not in type_of_each:
                type_of_each[label] = 1
            else:
                type_of_each[label] += 1
            if type_of_each[label] > maxim:
                maxim = type_of_each[label]

        oversample_factor = {}
        for key in type_of_each:
            oversample_factor[key] = maxim // type_of_each[key]

        samples = []
        for item in data:
            label = item["label"]
            samples += [item] * oversample_factor[label]

        random.shuffle(samples)

        print(type_of_each)

        return samples

    def __len__(self):
        return len(self.data_subset)

    def __getitem__(self, idx):
        item = self.data_subset[idx]
        guid = item["guid"]
        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]
        if self.transform:
            sentence1 = self.transform(sentence1)
            sentence2 = self.transform(sentence2)
        label = item["label"]
        return guid, sentence1, sentence2, label

    def load_data(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
