import torch
import torch.nn as nn
from torchmetrics import F1Score, Precision, Recall, Accuracy
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from transformers import AutoModel
import lightning.pytorch as pl
import json
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision(
    'high'
)

# define the LightningModule
class BertCurriculumClassifier(pl.LightningModule):
    def __init__(self, stats_folder: str):
        super().__init__()

        self.num_classes = 4
        self.stats_folder = stats_folder
        self.bert = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()

        self.fc1 = nn.Linear(768*2, 256)
        self.fc2 = nn.Linear(256, self.num_classes)

        self.train_stats = []
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

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


    def extract_bert_output(self, sentence):
        input_ids = torch.squeeze(sentence['input_ids'],dim= 1).to(device)
        attention_mask = torch.squeeze(sentence['attention_mask'],dim=1).to(device)

        with torch.no_grad():
            outputs = self.bert(input_ids = input_ids, attention_mask = attention_mask)
            cls_token = outputs[0][:,0,:]
            return cls_token
        

    def forward(self, batch, batch_idx):

        guid, sentence1,sentence2, labels = batch

        cls_token_1 = self.extract_bert_output(sentence1)
        cls_token_2 = self.extract_bert_output(sentence2)
        
        merged_cls = torch.cat((cls_token_1,cls_token_2),dim=1)

        x = self.fc1(merged_cls)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)     

        return guid, x,labels  

    def training_step(self, batch, batch_idx):

        guid, x,labels = self.forward(batch, batch_idx)
        loss = self.loss_fn(x, labels)
        input_max, max_indices = torch.max(x, dim=1)

        items = []
        for i in range(len(guid)):
            items.append({
                "guid": guid[i],
                f"logits_epoch_{self.current_epoch}": x[i].cpu().detach().numpy().tolist(),
                "gold": labels[i].cpu().detach().item()
            })

        self.train_stats += items

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    

    def on_train_epoch_end(self):
        with open(f'{self.stats_folder}/dynamics_epoch_{self.current_epoch}.jsonl', 'w', encoding="UTF-8") as f:
            json.dump(self.train_stats, f, indent=4, ensure_ascii=False)
        self.train_stats.clear()

    
    def validation_step(self, batch, batch_idx):
        
        _guid, x,labels = self.forward(batch, batch_idx)
        loss = self.loss_fn(x, labels)

        preds = torch.argmax(x, dim=1)

        self.accuracy(preds,labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        
        _guid, x,labels = self.forward(batch, batch_idx)
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

        _guid, x,_labels = self.forward(batch, batch_idx)
        preds = torch.argmax(x, dim=1)
        return preds




    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer