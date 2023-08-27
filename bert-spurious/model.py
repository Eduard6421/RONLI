from nli_learning.Dataset.NLIDataModule import NLIDataModule
from nli_learning.bert_spurious.bert_spurious import get_bert_embedding, train_bert_spurious

NUM_EPOCHS = 10

def train_pipeline(args):
    
    dataset_path = "dataset/datasets"
    print('Loading dataset')
    dataset = NLIDataModule(data_source_folder=dataset_path,dataset_name='bert_spurious_dataset', batch_size=256,transform=get_bert_embedding)
    print("Train BERT classifier")
    train_bert_spurious(datamodule=dataset, num_epochs=NUM_EPOCHS)