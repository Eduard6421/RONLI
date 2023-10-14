import os
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from nli_learning.Dataset.NLIDataset import NLIDataset
import lightning.pytorch as pl
import random

from nli_learning.Dataset.NLIDatasetOversample import NLIDatasetOversample


class NLIDataModuleOversample(pl.LightningDataModule):
    def __init__(self, data_source_folder, dataset_name, batch_size=64, transform=None):
        print("Initializing NLIDataModule Oversample")
        print(transform)

        super().__init__()
        self.batch_size = batch_size
        self.transform = transform

        # Source paths
        self.data_source_folder = data_source_folder

        self.train_source_data = os.path.join(self.data_source_folder, "train.json")
        self.val_source_data = os.path.join(self.data_source_folder, "validation.json")
        self.test_source_data = os.path.join(self.data_source_folder, "test.json")

        # Dataset paths
        self.train_path = os.path.join(
            self.data_source_folder, dataset_name, "train_dataset_oversample.pt"
        )
        self.val_path = os.path.join(
            self.data_source_folder, dataset_name, "validation_dataset_oversample.pt"
        )
        self.test_path = os.path.join(
            self.data_source_folder, dataset_name, "test_dataset_oversample.pt"
        )

    def prepare_data(self):
        if os.path.exists(self.train_path):
            print("Data was already preprepared")
            return

        print("Preparing the data")

        train_dataset = NLIDataset(
            self.train_source_data, self.batch_size, self.transform
        )
        val_dataset = NLIDataset(self.val_source_data, self.batch_size, self.transform)
        test_dataset = NLIDataset(
            self.test_source_data, self.batch_size, self.transform
        )

        print("Processing train dataset")
        train_items = []
        for item in train_dataset:
            train_items.append(item)

        print("Processing validation dataset")
        val_items = []
        for item in val_dataset:
            val_items.append(item)

        print("Processing test dataset")
        test_items = []
        for item in test_dataset:
            test_items.append(item)

        torch.save(train_items, self.train_path)
        torch.save(val_items, self.val_path)
        torch.save(test_items, self.test_path)

        print("Succesfully saved the datasets")

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        # Called on every process in DDP

        print("Loading Train Dataset")
        self.train_dataset = torch.load(self.train_path)

        print("Loading Validation Dataset")
        self.val_dataset = torch.load(self.val_path)

        print("Loading Test Dataset")
        self.test_dataset = torch.load(self.test_path)

    def _oversample(self, targets):
        # Precomputed oversampling

        oversample_factor = {
            0: 10,
            1: 20,
            2: 1,
            3: 1,
        }

        type_of_each = {}

        samples = []

        for item in targets:
            samples += [item] * oversample_factor[item[3]]
            if item[3] not in type_of_each:
                type_of_each[item[3]] = 1
            else:
                type_of_each[item[3]] += 1

        print(type_of_each)

        random.shuffle(samples)

        print(len(samples))

        for sample in samples:
            if len(sample) != 4:
                print(sample)
                raise Exception("Invalid sample")

        train_dataset = NLIDatasetOversample(samples, self.batch_size, self.transform)

        return train_dataset

    def train_dataloader(self):
        train_dataset = self._oversample(self.train_dataset)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
