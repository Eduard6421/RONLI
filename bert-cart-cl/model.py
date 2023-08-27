import numpy as np
from nli_learning.bert_curriculum_full.bert_curriculum_full import train_bert_curriculum_full

def train_pipeline(args):
    DATASET_PATH = "dataset/datasets"
    print("Train BERT classifier")
    train_bert_curriculum_full(dataset_path=DATASET_PATH)
