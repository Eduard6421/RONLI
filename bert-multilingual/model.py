import numpy as np
from nli_learning.bert_multilingual.BertMultilingualTrainingPipeline import bert_multilingual_train_pipeline

def train_pipeline(args):
    DATASET_PATH = "dataset/datasets"
    print("Train Multilingual Bert classifier")
    bert_multilingual_train_pipeline(dataset_path=DATASET_PATH)
    #bert_multilingual_zero_shot_train_pipeline(dataset_path=DATASET_PATH)
