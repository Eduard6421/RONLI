import numpy as np
from nli_learning.Dataset.NLIDataModuleCurriculum import NLIDataModuleCurriculum
from nli_learning.Dataset.NLIDataModuleSubset import NLIDataModuleSubset
from nli_learning.bert_curriculum_full.bert_curriculum_full import train_bert_curriculum_full

NUM_EPOCHS = 15

def bert_curriculum_full_train_pipeline(dataset_path):
    print("Train BERT classifier")
    train_bert_curriculum_full(dataset_path=dataset_path)
