import torch
import lightning.pytorch as pl

from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report

from nli_learning.gpt2.GPTClassificator import GPTClassificator

GPT_DIR = "gpt_checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('readerbench/RoGPT2-medium', padding_size='left')
print(tokenizer.all_special_tokens) # --> ['<|endoftext|>']
print(tokenizer.all_special_ids)    # --> [0]

eos_token = tokenizer.all_special_ids[0]

def get_gpt_embedding(sentence):
    encoding = tokenizer(sentence, max_length = 1023, truncation=True, return_tensors="pt")

    # Retrieving the position of the last token + 1 (the EOS token)
    final_token_position = encoding['input_ids'][0].shape[0]

    num_of_pads = 1024 - encoding['input_ids'][0].shape[0]
    
    encoding['input_ids'] = torch.cat((encoding['input_ids'][0], torch.full((num_of_pads,), eos_token, dtype=torch.long)), dim=0)
    encoding['attention_mask'] = torch.cat((encoding['attention_mask'][0], torch.zeros(num_of_pads, dtype=torch.long)), dim=0)

    # Setting the attention mask to 1 for the first EOS token
    encoding['attention_mask'][final_token_position] = 1

    return {
        'input_ids': encoding['input_ids'],
        'attention_mask' : encoding['attention_mask'],
        'eos_token_positions' : final_token_position
        }

def train_gpt(datamodule: pl.LightningDataModule, num_epochs: int):

    model = GPTClassificator()

    checkpoint_callback = ModelCheckpoint(dirpath = GPT_DIR, 
                                          filename = 'bert-{epoch:02d}-{val_loss:.2f}',
                                          monitor='val_loss',
                                          save_top_k=2)

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5)

    trainer = Trainer(devices=1, accelerator="gpu", default_root_dir=GPT_DIR, callbacks=[checkpoint_callback, early_stop_callback], max_epochs=num_epochs)
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, datamodule=datamodule)

    #predictions = trainer.predict(model, datamodule.test_dataloader())

    # transform array of tensors to a single tensor
    #predictions = torch.cat(predictions, dim=0)

    #labels = [_label for _sentence1,_sentence2,_label in datamodule.test_dataloader()]
    #labels = torch.cat(labels, dim=0)


    #print(predictions)
    #print(labels)

    #print(accuracy_score(predictions, labels))
    #print(classification_report(labels, predictions))

    trainer.fit(model=model,datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path='best')