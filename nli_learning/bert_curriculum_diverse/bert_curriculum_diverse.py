from sklearn.metrics import accuracy_score
import torch
from transformers import AutoTokenizer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import lightning.pytorch as pl

from nli_learning.bert.BertClassificator import BertClassificator
from sklearn.metrics import classification_report
from nli_learning.Dataset.NLIDataModuleCurriculumDiverse import (
    NLIDataModuleCurriculumDiverse,
)
from nli_learning.bert_curriculum_diverse.BertCurriculumDiverseClassifier import (
    BertCurriculumDiverseClassifier,
)

BERT_DIR = "bert_checkpoints/rocurriculumdiverse"
BERT_STATS = "bert_stats/rocurriculumdiverse"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "dumitrescustefan/bert-base-romanian-cased-v1"
)

# print(tokenizer.all_special_tokens) # --> ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
# print(tokenizer.all_special_ids)    # --> [1, 3, 0, 2, 4]


def get_bert_embedding(sentence: str):
    sentence = (
        sentence.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
    )
    encoding = tokenizer(
        sentence,
        add_special_tokens=True,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
    }


def train_bert_cart_stra_cl_plus(dataset_path: str):
    # model
    model = BertCurriculumDiverseClassifier(stats_folder=BERT_STATS)
    checkpoint_callback = ModelCheckpoint(
        dirpath=BERT_DIR,
        every_n_train_steps=100,
        filename="stage-0-bert-{step:02d}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=-1,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=3)

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        default_root_dir=BERT_DIR,
        callbacks=[checkpoint_callback],
        max_steps=100,
        val_check_interval=100,
        check_val_every_n_epoch=None,
    )

    module_1 = NLIDataModuleCurriculumDiverse(
        train_stage=0,
        data_source_folder=dataset_path,
        dataset_name="bert_curriculum_diverse_dataset",
        batch_size=256,
        transform=get_bert_embedding,
    )

    trainer.fit(model=model, datamodule=module_1)

    checkpoint_callback = ModelCheckpoint(
        dirpath=BERT_DIR,
        every_n_train_steps=100,
        filename="stage-1-bert-{step:02d}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=-1,
    )

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        default_root_dir=BERT_DIR,
        callbacks=[checkpoint_callback],
        max_steps=200,
        val_check_interval=100,
        check_val_every_n_epoch=None,
    )

    module_2 = NLIDataModuleCurriculumDiverse(
        train_stage=1,
        data_source_folder=dataset_path,
        dataset_name="bert_curriculum_diverse_dataset",
        batch_size=256,
        transform=get_bert_embedding,
    )
    trainer.fit(model=model, datamodule=module_2)

    checkpoint_callback = ModelCheckpoint(
        dirpath=BERT_DIR,
        every_n_train_steps=100,
        filename="stage-2-bert-{step:02d}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=-1,
    )

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        default_root_dir=BERT_DIR,
        callbacks=[checkpoint_callback],
        max_steps=300,
        val_check_interval=100,
        check_val_every_n_epoch=None,
    )

    module_3 = NLIDataModuleCurriculumDiverse(
        train_stage=2,
        data_source_folder=dataset_path,
        dataset_name="bert_curriculum_diverse_dataset",
        batch_size=256,
        transform=get_bert_embedding,
    )
    trainer.fit(model=model, datamodule=module_3)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=BERT_DIR,
        every_n_train_steps=100,
        filename="stage-3-bert-{step:02d}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=-1,
    )

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        default_root_dir=BERT_DIR,
        callbacks=[checkpoint_callback],
        max_steps=400,
        val_check_interval=100,
        check_val_every_n_epoch=None,
    )

    module_4 = NLIDataModuleCurriculumDiverse(
        train_stage=3,
        data_source_folder=dataset_path,
        dataset_name="bert_curriculum_diverse_dataset",
        batch_size=256,
        transform=get_bert_embedding,
    )
    trainer.fit(model=model, datamodule=module_4)    

    
    checkpoint_callback = ModelCheckpoint(
        dirpath=BERT_DIR,
        every_n_train_steps=100,
        filename="stage-2-bert-{step:02d}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=-1,
    )

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        default_root_dir=BERT_DIR,
        callbacks=[checkpoint_callback],
        max_steps=1500,
        val_check_interval=100,
        check_val_every_n_epoch=None,
    )

    module_5 = NLIDataModuleCurriculumDiverse(
        train_stage=4,
        data_source_folder=dataset_path,
        dataset_name="bert_curriculum_diverse_dataset",
        batch_size=256,
        transform=get_bert_embedding,
    )
    trainer.fit(model=model, datamodule=module_5)    
    trainer.test(model=model, datamodule=module_5, ckpt_path="best")
