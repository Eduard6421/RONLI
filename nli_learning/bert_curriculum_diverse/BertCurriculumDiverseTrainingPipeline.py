import numpy as np
from nli_learning.bert_curriculum_diverse.bert_curriculum_diverse import (
    get_bert_embedding,
    train_bert_cart_stra_cl_plus,
)

NUM_EPOCHS = 15


def bert_curriculum_diverse_train_pipeline(dataset_path):
    print("Train BERT classifier on diverse curriculum")
    train_bert_cart_stra_cl_plus(dataset_path=dataset_path)
