import numpy as np
from nli_learning.bert_multilingual.bert_multilingual import get_bert_embedding, train_bert_multilingual

NUM_EPOCHS = 15

def bert_multilingual_train_pipeline(dataset_path):
    print("Train BERT Multilingual classifier")
    train_bert_multilingual(dataset_path=dataset_path)
