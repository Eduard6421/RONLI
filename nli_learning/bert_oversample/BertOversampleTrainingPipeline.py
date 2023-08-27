import numpy as np

from nli_learning.bert_oversample.bert_oversample import train_bert_oversample

NUM_EPOCHS = 10

def bert_oversample_train_pipeline(dataset_path):
    print("Train BERT Multilingual classifier")
    train_bert_oversample(dataset_path=dataset_path)
