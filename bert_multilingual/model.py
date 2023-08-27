import numpy as np
from nli_learning.bert_multilingual.bert_multilingual import train_bert_multilingual

def train_pipeline(args):
    DATASET_PATH = "dataset/datasets"
    print("Train Multilingual Bert classifier")
    train_bert_multilingual(dataset_path=DATASET_PATH)
