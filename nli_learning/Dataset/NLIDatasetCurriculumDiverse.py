import json

from torch.utils.data import Dataset
from torch.utils.data import random_split


class NLIDatasetCurriculumDiverse(Dataset):
    def __init__(self, data, batch_size=8, transform=None):
        self.data = data
        self.batch_size = batch_size
        self.transform = transform
        self.data_subset = self.data[: self.unlocked_lengths[self.current_unlock_index]]
        self.data_subset = self._oversample(self.data_subset)

    def _oversample(self, data):
        # Precomputed oversampling

        type_of_each = {}

        maxim = 0
        for item in data:
            if item[3] not in type_of_each:
                type_of_each[item[3]] = 1
            else:
                type_of_each[item[3]] += 1
            if type_of_each[item[3]] > maxim:
                maxim = type_of_each[item[3]]

        oversample_factor = {}
        for key in type_of_each:
            oversample_factor[key] = maxim // type_of_each[key]

        samples = []
        for item in data:
            samples += [item] * oversample_factor[item[3]]

        random.shuffle(samples)

        print(type_of_each)

        return samples

    def __len__(self):
        return len(self.data_subset)

    def __getitem__(self, idx):
        item = self.data[idx]
        guid = item["guid"]
        sentence1 = item["sentence1"]
        sentence2 = item["sentence2"]
        if self.transform:
            sentence1 = self.transform(sentence1)
            sentence2 = self.transform(sentence2)
        label = item["label"]
        return guid, sentence1, sentence2, label
