import torch
import torch.nn as nn
from torchmetrics import F1Score, Precision, Recall, Accuracy
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from transformers import AutoModelForCausalLM
import lightning.pytorch as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision(
    'high'
)

# define the LightningModule
class GPTClassificator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.gpt = AutoModelForCausalLM.from_pretrained('readerbench/RoGPT2-medium')

        for param in self.gpt.parameters():
            param.requires_grad = False


        self.num_classes = 4

        self.fc1 = nn.Linear(50257*2, 256)
        self.fc2 = nn.Linear(256, self.num_classes)
        
        self.relu = nn.ReLU()


        self.loss_fn = nn.CrossEntropyLoss()

        self.f1_micro = F1Score(task="multiclass", num_classes=self.num_classes, average="micro")
        self.f1_macro = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.f1_each  = F1Score(task="multiclass", num_classes=self.num_classes, average=None )

        self.precision_micro = Precision(task="multiclass", num_classes=self.num_classes, average="micro")
        self.precision_macro = Precision(task="multiclass", num_classes=self.num_classes, average="macro")
        self.precision_each  = Precision(task="multiclass", num_classes=self.num_classes, average=None )

        self.recall_micro = Recall(task="multiclass", num_classes=self.num_classes, average="micro")
        self.recall_macro = Recall(task="multiclass", num_classes=self.num_classes, average="macro")
        self.recall_each  = Recall(task="multiclass", num_classes=self.num_classes, average=None )

        self.accuracy = Accuracy('multiclass', num_classes=self.num_classes)


    def extract_gpt_output(self, sentence):
        input_ids = torch.squeeze(sentence['input_ids'],dim= 1)
        attention_mask = torch.squeeze(sentence['attention_mask'],dim=1)
        eos_token_position = sentence['eos_token_positions']

        with torch.no_grad():
            outputs = self.gpt(input_ids = input_ids, attention_mask = attention_mask)
            cls_token = outputs[0][torch.arange(outputs[0].size(0)), eos_token_position]
            return cls_token

    def training_step(self, batch, batch_idx):


        guid,sentence1,sentence2,labels = batch

        cls_token_1 = self.extract_gpt_output(sentence1)
        cls_token_2 = self.extract_gpt_output(sentence2)
        
        merged_cls = torch.cat((cls_token_1,cls_token_2),dim=1)


        x = self.fc1(merged_cls)
        x = self.relu(x)
        x = self.fc2(x)

        loss = self.loss_fn(x, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        guid,sentence1,sentence2,labels = batch

        cls_token_1 = self.extract_gpt_output(sentence1)
        cls_token_2 = self.extract_gpt_output(sentence2)

        merged_cls = torch.cat((cls_token_1,cls_token_2),dim=1)

        x = self.fc1(merged_cls)
        x = self.relu(x)
        x = self.fc2(x)


        loss = self.loss_fn(x, labels)
        preds = torch.argmax(x, dim=1)

        self.accuracy(preds,labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        guid,sentence1,sentence2,labels = batch

        cls_token_1 = self.extract_gpt_output(sentence1)
        cls_token_2 = self.extract_gpt_output(sentence2)
        merged_cls = torch.cat((cls_token_1,cls_token_2),dim=1)

        x = self.fc1(merged_cls)
        x = self.relu(x)
        x = self.fc2(x)

        loss = self.loss_fn(x, labels)
        preds = torch.argmax(x, dim=1)

        self.accuracy(preds,labels)

        self.precision_micro(preds, labels)
        self.precision_macro(preds, labels) 
        class_precisions = self.precision_each(preds, labels)

        self.recall_micro(preds, labels)
        self.recall_macro(preds, labels)
        class_recalls = self.recall_each(preds, labels)

        self.f1_micro(preds, labels)
        self.f1_macro(preds, labels)
        class_f1 = self.f1_each(preds, labels)

        for i in range(self.num_classes):
            self.log(f'precision_{i}', class_precisions[i], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'recall_{i}', class_recalls[i], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'f1_{i}',class_f1[i], on_step=False, on_epoch=True, prog_bar=True)



        self.log('precision_micro', self.precision_micro, on_step=False, on_epoch=True, prog_bar=True)
        self.log('precision macro', self.precision_macro, on_step=False, on_epoch=True, prog_bar=True)

        self.log('recall_micro', self.recall_micro, on_step=False, on_epoch=True, prog_bar=True)
        self.log('recall_macro', self.recall_macro, on_step=False, on_epoch=True, prog_bar=True)

        self.log('f1_micro', self.f1_micro, on_step=False, on_epoch=True, prog_bar=True)
        self.log('f1_macro', self.f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_accuracy', self.accuracy , on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        guid,sentence1,sentence2,labels = batch

        cls_token_1 = self.extract_gpt_output(sentence1)
        cls_token_2 = self.extract_gpt_output(sentence2)
        merged_cls = torch.cat((cls_token_1,cls_token_2),dim=1)


        x = self.fc1(merged_cls)
        x = self.relu(x)
        x = self.fc2(x)

        preds = torch.argmax(x, dim=1)


        return preds




    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer