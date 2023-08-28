from sklearn.metrics import accuracy_score
import torch
from transformers import AutoTokenizer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import lightning.pytorch as pl

from nli_learning.bert.BertClassificator import BertClassificator
from sklearn.metrics import classification_report

BERT_DIR = "bert_checkpoints/robase"
BERT_STATS = "bert_stats/robase"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

#print(tokenizer.all_special_tokens) # --> ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
#print(tokenizer.all_special_ids)    # --> [1, 3, 0, 2, 4]

def get_bert_embedding(sentence: str):
    sentence = sentence.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
    encoding = tokenizer(sentence, add_special_tokens=True,padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask' : encoding['attention_mask']
        }


def train_bert(datamodule: pl.LightningDataModule, num_epochs: int):

    # model
    model = BertClassificator(stats_folder=BERT_STATS)#.load_from_checkpoint("bert_checkpoints/bert-epoch=04-val_loss=0.30.ckpt")


    checkpoint_callback = ModelCheckpoint(dirpath = BERT_DIR, 
                                          filename = 'bert-{epoch:02d}-{val_loss:.2f}',
                                          monitor='val_loss',
                                          save_top_k=3)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=8)

    trainer = Trainer(devices=1, 
                      accelerator="gpu", default_root_dir=BERT_DIR, callbacks=[checkpoint_callback, early_stop_callback], max_epochs=num_epochs)
    #tuner = Tuner(trainer)
    #tuner.scale_batch_size(model, datamodule=datamodule)

    trainer.fit(model=model,datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path='best')