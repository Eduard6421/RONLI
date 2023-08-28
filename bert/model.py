import numpy as np
from nli_learning.Dataset.NLIDataModule import NLIDataModule
from nli_learning.bert.bert_ro import get_bert_embedding, train_bert

NUM_EPOCHS = 10

def train_pipeline(args):

    dataset_path = "dataset/datasets"
    print('Loading dataset')
    dataset = NLIDataModule(data_source_folder=dataset_path,dataset_name='bert_dataset', batch_size=128,transform=get_bert_embedding)
    train_bert(datamodule=dataset, num_epochs=NUM_EPOCHS)
