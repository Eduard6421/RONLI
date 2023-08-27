from sklearn.metrics import accuracy_score
import torch
from transformers import AutoTokenizer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import lightning.pytorch as pl

from nli_learning.bert.BertClassificator import BertClassificator
from sklearn.metrics import classification_report

from nli_learning.bert_multilingual_zero_shot.BertMultilingualZeroShotClassifier import BertMultilingualZeroShotClassifier

BERT_DIR = "bert_checkpoints/bertmultilingualzeroshot"
BERT_STATS = "bert_stats/bertmultilingualzeroshot"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

#print(tokenizer.all_special_tokens) # --> ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
#print(tokenizer.all_special_ids)    # --> [1, 3, 0, 2, 4]

def get_bert_embedding(sentence: str):
    #sentence = sentence.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
    encoding = tokenizer(sentence, add_special_tokens=True,padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask' : encoding['attention_mask']
        }


def train_bert_multilingual_zero_shot(datamodule: pl.LightningDataModule, num_epochs: int):

    # model
    model = BertMultilingualZeroShotClassifier(stats_folder=BERT_STATS)#.load_from_checkpoint("bert_checkpoints/bert-epoch=04-val_loss=0.30.ckpt")

    trainer = Trainer(devices=1, 
                      accelerator="gpu", default_root_dir=BERT_DIR, callbacks=[], max_epochs=num_epochs)
    #tuner = Tuner(trainer)
    #tuner.scale_batch_size(model, datamodule=datamodule)

    #predictions = trainer.predict(model, datamodule.test_dataloader())

    # transform array of tensors to a single tensor
    #predictions = torch.cat(predictions, dim=0)

    #labels = [_label for _sentence1,_sentence2,_label in datamodule.test_dataloader()]
    #labels = torch.cat(labels, dim=0)


    #print(predictions)
    #print(labels)

    #print(accuracy_score(predictions, labels))
    #print(classification_report(labels, predictions))


    #trainer.fit(model=model,datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)