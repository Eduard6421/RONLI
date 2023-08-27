from sklearn.metrics import accuracy_score
import torch
from transformers import AutoTokenizer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import lightning.pytorch as pl
from nli_learning.Dataset.NLIDataModule import NLIDataModule
from nli_learning.Dataset.NLIDataModuleCurriculum import NLIDataModuleCurriculum
from nli_learning.Dataset.NLIDataModuleMultiLanguage import NLIDataModuleMultilanguage

from nli_learning.bert.BertClassificator import BertClassificator
from sklearn.metrics import classification_report

from nli_learning.bert_curriculum.BertCurriculumClassifier import BertCurriculumClassifier
from nli_learning.bert_multilingual.BertMultilingualClassifier import BertMultilingualClassifier

BERT_DIR = "bert_checkpoints/bertmultilingual"
BERT_STATS = "bert_stats/bertmultilingual"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

#print(tokenizer.all_special_tokens) # --> ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
#print(tokenizer.all_special_ids)    # --> [1, 3, 0, 2, 4]

def get_bert_embedding(sentence: str):
    encoding = tokenizer(sentence, add_special_tokens=True,padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask' : encoding['attention_mask']
        }


def train_bert_multilingual(dataset_path: str):

    # model
    model = BertMultilingualClassifier(stats_folder=BERT_STATS)

    checkpoint_callback = ModelCheckpoint(dirpath = BERT_DIR, 
                                          every_n_train_steps = 51,
                                          filename = 'stage-0-bert-{step:02d}-{val_loss:.2f}',
                                          monitor='val_loss',
                                          save_top_k=-1)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=3)

    trainer = Trainer(devices=1, 
                      accelerator="gpu", default_root_dir=BERT_DIR, callbacks=[checkpoint_callback, early_stop_callback],
                      max_steps=228,
                      val_check_interval=50,
                      check_val_every_n_epoch=None)
    
    print('Loading dataset')
    dataset = NLIDataModuleMultilanguage(data_source_folder=dataset_path,dataset_name='bert_dataset', batch_size=128,transform=get_bert_embedding)

    trainer.fit(model=model,datamodule=dataset)

    # You can also remove early stopping from the experiments and just select the best checkpoint
    trainer.test(model=model, datamodule=dataset, ckpt_path='best')