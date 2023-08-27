import numpy as np
from nli_learning.bert_oversample.BertOversampleTrainingPipeline import bert_oversample_train_pipeline

def train_pipeline(args):
    DATASET_PATH = "dataset/datasets"
    print("Train RoBert Oversample")
    bert_oversample_train_pipeline(dataset_path=DATASET_PATH)
