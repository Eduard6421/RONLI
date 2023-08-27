import numpy as np
from nli_learning.bert_curriculum.BertCurriculumTrainingPipeline import bert_curriculum_train_pipeline

def train_pipeline(args):
    DATASET_PATH = "dataset/datasets"
    print("Train BERT classifier")
    bert_curriculum_train_pipeline(dataset_path=DATASET_PATH)