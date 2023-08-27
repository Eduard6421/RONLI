import numpy as np
from nli_learning.bert_curriculum_diverse.bert_curriculum_diverse import train_bert_curriculum_diverse

def train_pipeline(args):
    DATASET_PATH = "dataset/datasets"
    print("Train BERT classifier")
    train_bert_curriculum_diverse(dataset_path=DATASET_PATH)