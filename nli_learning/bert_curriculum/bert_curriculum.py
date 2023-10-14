from sklearn.metrics import accuracy_score
import torch
from transformers import AutoTokenizer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import lightning.pytorch as pl
from nli_learning.Dataset.NLIDataModuleCurriculum import NLIDataModuleCurriculum

from nli_learning.bert.BertClassificator import BertClassificator
from sklearn.metrics import classification_report

from nli_learning.bert_curriculum.BertCurriculumClassifier import (
    BertCurriculumClassifier,
)

BERT_DIR = "bert_checkpoints/rocurriculum"
BERT_STATS = "bert_stats/rocurriculum"

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


def train_bert_curriculum(dataset_path: str):
    # This subset of data consists of approx 58k items ( easy / ambiguous / hard with duplicate removal)
    # However, we consider the data oversampled before the curriculum learning
    # 2952  contrastive becomes: 2592 * 10 = 25920
    # 1300  entailment  becomes: 1300 * 21 = 27300
    # 25722 causal  becomes: 25722 * 1 = 25722
    # 28500 neutral becomes: 28500 * 1 = 28500
    # Total: 107442

    # We have batches of 512 items
    # In each epoch ther are 210 batches
    # We train for 10 epochs in the original bert, meaning that we train on 2100 iterations.
    # We try to add into the curriculum all the data before the halfway of iterations
    # First entry at (2100/2)/6.
    # Second entry at (2100/2)/6*2.
    # Third entry at 2100/2 + (2100/2)/6*3.
    # Afterwards just train until validation reflects overfit.

    print("RUNNING WITH TRAINIGN STEPS")

    # model
    model = BertCurriculumClassifier(stats_folder=BERT_STATS)

    checkpoint_callback = ModelCheckpoint(
        dirpath=BERT_DIR,
        every_n_train_steps=101,
        filename="stage-0-bert-{step:02d}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=-1,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10)

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        default_root_dir=BERT_DIR,
        callbacks=[checkpoint_callback],
        max_steps=175,
        val_check_interval=100,
        check_val_every_n_epoch=None,
    )

    print("Loading dataset")
    module_1 = NLIDataModuleCurriculum(
        train_stage=0,
        data_source_folder=dataset_path,
        dataset_name="bert_curriculum_dataset",
        batch_size=512,
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
        max_steps=175 * 2,
        val_check_interval=100,
        check_val_every_n_epoch=None,
    )

    module_2 = NLIDataModuleCurriculum(
        train_stage=1,
        data_source_folder=dataset_path,
        dataset_name="bert_curriculum_dataset",
        batch_size=512,
        transform=get_bert_embedding,
    )
    trainer.fit(model=model, datamodule=module_2)

    checkpoint_callback = ModelCheckpoint(
        dirpath=BERT_DIR,
        every_n_train_steps=51,
        filename="stage-2-bert-{step:02d}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=-1,
    )

    trainer = Trainer(
        devices=1,
        accelerator="gpu",
        default_root_dir=BERT_DIR,
        callbacks=[checkpoint_callback, early_stop_callback],
        max_steps=175 * 3 + 175 * 6,
        val_check_interval=100,
        check_val_every_n_epoch=None,
    )

    module_3 = NLIDataModuleCurriculum(
        train_stage=2,
        data_source_folder=dataset_path,
        dataset_name="bert_curriculum_dataset",
        batch_size=512,
        transform=get_bert_embedding,
    )
    trainer.fit(model=model, datamodule=module_3)

    # Here try to select the best checkpoint across stages rather just the one from this stage.
    # Here you can experiment with various stages of training
    # You can also remove early stopping from the experiments and just select the best checkpoint
    trainer.test(model=model, datamodule=module_3, ckpt_path="best")
