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
from nli_learning.Dataset.NLIDataModuleOversample import NLIDataModuleOversample

from nli_learning.bert.BertClassificator import BertClassificator
from sklearn.metrics import classification_report

from nli_learning.bert_curriculum.BertCurriculumClassifier import BertCurriculumClassifier
from nli_learning.bert_multilingual.BertMultilingualClassifier import BertMultilingualClassifier
from nli_learning.bert_oversample.BertOversampleClassifier import BertOversampleClassifier

BERT_DIR = "bert_checkpoints/bertoversample"
BERT_STATS = "bert_stats/bertoversample"

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


def train_bert_oversample(dataset_path: str):

    # model
    model = BertOversampleClassifier(stats_folder=BERT_STATS)

    checkpoint_callback = ModelCheckpoint(dirpath = BERT_DIR, 
                                          every_n_train_steps = 500,
                                          filename = 'stage-0-bert-{step:02d}-{val_loss:.2f}',
                                          monitor='val_loss',
                                          save_top_k=-1)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=3)

    trainer = Trainer(devices=1, 
                      accelerator="gpu", default_root_dir=BERT_DIR, callbacks=[checkpoint_callback, early_stop_callback], max_epochs=15)
    
    print('Loading dataset')
    datamodule = NLIDataModuleOversample(data_source_folder=dataset_path,dataset_name='bert_dataset', batch_size=256,transform=get_bert_embedding)


    #tuner = Tuner(trainer)
    #tuner.scale_batch_size(model, datamodule=datamodule)

    trainer.fit(model=model,datamodule=datamodule)

    # You can also remove early stopping from the experiments and just select the best checkpoint
    trainer.test(model=model, datamodule=datamodule, ckpt_path='best')