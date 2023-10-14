import numpy as np
from nli_learning.bert_curriculum.bert_curriculum import (
    get_bert_embedding,
    train_bert_curriculum,
)

NUM_EPOCHS = 15


def train_bert_cart_cl_plus(dataset_path):
    print("Train BERT classifier")
    train_bert_curriculum(dataset_path=dataset_path)