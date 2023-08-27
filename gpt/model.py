import numpy as np
from nli_learning.Dataset.NLIDataModule import NLIDataModule
from nli_learning.gpt2.gpt_ro import get_gpt_embedding, train_gpt

NUM_EPOCHS = 10

def train_pipeline(args):
    dataset_path = "dataset/datasets"
    dataset = NLIDataModule(data_source_folder=dataset_path,dataset_name='gpt_dataset', batch_size=1, transform=get_gpt_embedding)
    print("Train GPT classifier")
    train_gpt(dataset, NUM_EPOCHS)
