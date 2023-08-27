import numpy as np
from nli_learning.Dataset.NLIDataModuleSubset import NLIDataModuleSubset
from nli_learning.bert_subset.bert_subset import get_bert_embedding, train_bertsubset

NUM_EPOCHS = 15

def bert_subset_train_pipeline(dataset_path, dataset_subset):


    print('Loading dataset')
    dataset = NLIDataModuleSubset(data_source_folder=dataset_path, subset = dataset_subset, dataset_name='bert_dataset', batch_size=256,transform=get_bert_embedding)
    dataset.prepare_data()
    dataset.setup()

    print("Train BERT classifier")
    train_bertsubset(datamodule=dataset, num_epochs=NUM_EPOCHS, dataset_subset = dataset_subset)
