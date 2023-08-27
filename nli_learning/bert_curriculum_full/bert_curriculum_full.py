from sklearn.metrics import accuracy_score
import torch
from transformers import AutoTokenizer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import lightning.pytorch as pl
from nli_learning.Dataset.NLIDataModuleCurriculumFull import NLIDataModuleCurriculumFull

from sklearn.metrics import classification_report

from nli_learning.bert_curriculum.BertCurriculumClassifier import BertCurriculumClassifier

BERT_DIR = "bert_checkpoints/rocurriculumcart"
BERT_STATS = "bert_stats/rocurriculumcart"

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


def train_bert_curriculum_full(dataset_path: str):

    # model
    model = BertCurriculumClassifier(stats_folder=BERT_STATS)

    # This subset of data consists of approx 26k items ( easy / ambiguous / hard with duplicate removal)
    # We have batches of 256 items
    # In each epoch there are 101 batches
    # We train for 10 epochs in original bert mearning that we train on 1010 iterations
    # We try to add into the curriculum all the data before the halfway of iterations
    # First entry at (1010/2)/3.
    # Second entry at (1010/2)/3*2.
    # Third entry at (1010/2)/3*3.
    # Afterwards just train until validation reflects overfit.

    checkpoint_callback = ModelCheckpoint(dirpath = BERT_DIR, 
                                          every_n_train_steps = 51,
                                          filename = 'stage-0-bert-{step:02d}-{val_loss:.2f}',
                                          monitor='val_loss',
                                          save_top_k=-1)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5)

    trainer = Trainer(devices=1, 
                      accelerator="gpu", default_root_dir=BERT_DIR, callbacks=[checkpoint_callback],
                      max_steps=168,
                      val_check_interval=50,
                      check_val_every_n_epoch=None)
    
    module_1 = NLIDataModuleCurriculumFull(train_stage=0, data_source_folder=dataset_path, dataset_name='bert_curriculum_full_dataset', batch_size=256,transform=get_bert_embedding)
    trainer.fit(model=model,datamodule=module_1)

    
    checkpoint_callback = ModelCheckpoint(dirpath = BERT_DIR, 
                                          every_n_train_steps = 51,
                                          filename = 'stage-1-bert-{step:02d}-{val_loss:.2f}',
                                          monitor='val_loss',
                                          save_top_k=-1)    
    
    trainer = Trainer(devices=1, 
                      accelerator="gpu", default_root_dir=BERT_DIR, callbacks=[checkpoint_callback],
                      max_steps=168,
                      val_check_interval=50,
                      check_val_every_n_epoch=None)


    module_2 = NLIDataModuleCurriculumFull(train_stage=1, data_source_folder=dataset_path, dataset_name='bert_curriculum_full_dataset', batch_size=256,transform=get_bert_embedding)
    trainer.fit(model=model,datamodule=module_2)

    checkpoint_callback = ModelCheckpoint(dirpath = BERT_DIR, 
                                          every_n_train_steps = 51,
                                          filename = 'stage-2-bert-{step:02d}-{val_loss:.2f}',
                                          monitor='val_loss',
                                          save_top_k=-1)    
    
    # Train for the rest of the iterations
    trainer = Trainer(devices=1, 
                      accelerator="gpu", default_root_dir=BERT_DIR, callbacks=[checkpoint_callback],
                      max_steps=2000,
                      val_check_interval=50,
                      check_val_every_n_epoch=None)


    # Here try to select the best checkpoint across stages rather just the one from this stage.
    # Here you can experiment with various stages of training
    # You can also remove early stopping from the experiments and just select the best checkpoint
    
    module_3 = NLIDataModuleCurriculumFull(train_stage=2, data_source_folder=dataset_path, dataset_name='bert_curriculum_full_dataset', batch_size=256,transform=get_bert_embedding)
    trainer.fit(model=model,datamodule=module_3)
    trainer.test(model=model, datamodule=module_3, ckpt_path='best')