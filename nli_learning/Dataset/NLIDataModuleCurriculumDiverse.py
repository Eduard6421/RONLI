import os
import torch

from torch.utils.data import DataLoader
from nli_learning.Dataset.NLIDataset import NLIDataset
import lightning.pytorch as pl

from nli_learning.Dataset.NLIDatasetCurriculumDiverse import NLIDatasetCurriculumDiverse

class NLIDataModuleCurriculumDiverse(pl.LightningDataModule):

    def __init__(self, data_source_folder, dataset_name, batch_size = 64, train_stage = 0, transform = None):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transform
        self.train_stage = train_stage
        self.fractions = [0.2, 0.4, 0.6, 0.8, 1]

        # Source paths
        self.data_source_folder = data_source_folder

        self.train_source_data = os.path.join(self.data_source_folder, 'train_curriculum.json')
        self.val_source_data = os.path.join(self.data_source_folder, 'validation.json')
        self.test_source_data = os.path.join(self.data_source_folder, 'test.json')

        # Dataset paths
        self.train_path = os.path.join(self.data_source_folder, dataset_name,'train_dataset.pt')
        self.val_path = os.path.join(self.data_source_folder, dataset_name,'validation_dataset.pt')
        self.test_path = os.path.join(self.data_source_folder, dataset_name,'test_dataset.pt')

    def prepare_data(self):

        
        import json 
        
        with open(self.train_source_data,'r',encoding='UTF-8') as f:
            data = json.load(f)
            
        label_0 = [item for item in data if item['label'] == 0]
        label_1 = [item for item in data if item['label'] == 1]
        label_2 = [item for item in data if item['label'] == 2]
        label_3 = [item for item in data if item['label'] == 3]
        
        label_0 = label_0[:int(len(label_0)*self.fractions[self.train_stage])]
        label_1 = label_1[:int(len(label_1)*self.fractions[self.train_stage])]
        label_2 = label_2[:int(len(label_2)*self.fractions[self.train_stage])]
        label_3 = label_3[:int(len(label_3)*self.fractions[self.train_stage])]
        
        train_set = label_0 + label_1 + label_2 + label_3
        
        train_dataset = NLIDatasetCurriculumDiverse(train_set, transform = self.transform)
        val_dataset   = NLIDataset(self.val_source_data, self.batch_size, self.transform)
        test_dataset  = NLIDataset(self.test_source_data, self.batch_size, self.transform)
        
        train_items = []
        for item in train_dataset:
            train_items.append(item)

        val_items = []
        for item in val_dataset:
            val_items.append(item)

        test_items = []
        for item in test_dataset:
            test_items.append(item)
            
        torch.save(train_items, self.train_path)
        torch.save(val_items, self.val_path)
        torch.save(test_items, self.test_path)


    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        # Called on every process in DDP

        #print('Loading Train Dataset')
        #self.train_dataset = torch.load(self.train_path)
        
        print("Setting up data module")
        self.train_dataset = torch.load(self.train_path)
        print(len(self.train_dataset))
        
        self.val_dataset = torch.load(self.val_path)
        self.test_dataset = torch.load(self.test_path)


    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,shuffle=False)