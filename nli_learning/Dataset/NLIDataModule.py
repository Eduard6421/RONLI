import os
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from nli_learning.Dataset.NLIDataset import NLIDataset
import lightning.pytorch as pl

class NLIDataModule(pl.LightningDataModule):

    def __init__(self, data_source_folder, dataset_name, batch_size = 64, transform = None):

        print('Initializing NLIDataModule')
        print(transform)

        super().__init__()
        self.batch_size = batch_size
        self.transform = transform

        # Source paths
        self.data_source_folder = data_source_folder

        self.train_source_data = os.path.join(self.data_source_folder, 'train.json')
        self.val_source_data = os.path.join(self.data_source_folder, 'validation.json')
        self.test_source_data = os.path.join(self.data_source_folder, 'test.json')

        # Dataset paths
        self.train_path = os.path.join(self.data_source_folder, dataset_name,'train_dataset.pt')
        self.val_path = os.path.join(self.data_source_folder, dataset_name,'validation_dataset.pt')
        self.test_path = os.path.join(self.data_source_folder, dataset_name,'test_dataset.pt')

    def prepare_data(self):

        #if(os.path.exists(self.train_path)):
        #    print('Data was already preprepared')
        #    return

        print('Preparing the data')
        
        train_dataset = NLIDataset(self.train_source_data, self.batch_size, self.transform)
        val_dataset   = NLIDataset(self.val_source_data, self.batch_size, self.transform)
        test_dataset  = NLIDataset(self.test_source_data, self.batch_size, self.transform)
        
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

        print('Loading Train Dataset')
        self.train_dataset = torch.load(self.train_path)

        print('Loading Validation Dataset')
        self.val_dataset = torch.load(self.val_path)

        print('Loading Test Dataset')
        self.test_dataset = torch.load(self.test_path)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,shuffle=False)