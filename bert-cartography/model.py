import numpy as np
from nli_learning.Dataset.NLIDataModuleSubset import NLIDataModuleSubset
from nli_learning.bert_subset.bert_subset import get_bert_embedding, train_bertsubset

NUM_EPOCHS = 15

def train_pipeline(args):

    DATASET_PATH = "dataset/datasets"
    print('Loading dataset')
    dataset = NLIDataModuleSubset(data_source_folder=DATASET_PATH, subset = args.subset, dataset_name='bert_dataset', batch_size=256,transform=get_bert_embedding)
    print("Train BERT classifier")
    train_bertsubset(datamodule=dataset, num_epochs=NUM_EPOCHS, dataset_subset = args.subset)
