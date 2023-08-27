import numpy as np
import torch
import torch.nn as nn
from torchmetrics import F1Score, Precision, Recall, Accuracy
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from transformers import AutoModel
import lightning.pytorch as pl
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision(
    'high'
)

# define the LightningModule
class BertMultilingualZeroShotClassifier(pl.LightningModule):
    def __init__(self, stats_folder: str):
        super().__init__()

        self.num_classes = 4
        self.stats_folder = stats_folder
        self.bert = AutoModel.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        self.tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

        self.classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",device=0)

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
        
    def forward(self, batch, batch_idx):

        guid, sentence1,sentence2, labels = batch
        label_names = ["contrastive", "entailment", "consequential","neutral"]

        label_to_id = {
            "contrastive": 0,
            "entailment": 1,
            "consequential": 2,
            "neutral": 3
        }

        parsed_inputs = []

        for i in range(len(sentence1)):
            full_sentence = sentence1[i] + " " + sentence2[i]
            parsed_inputs.append(full_sentence)
            #output = self.classifier(full_sentence, label_names, multi_label=False)
            #result = label_to_id[output['labels'][0]]
            #temp_preds = [0,0,0,0]
            #temp_preds[result] = 1
            #preds.append(temp_preds)
            #print(i)


        parsed_inputs = np.array(parsed_inputs)
        label_names = np.array(label_names)

        

        outputs = []

        for idx,item in enumerate(parsed_inputs):
            output = self.classifier(item, label_names, multi_label=False)
            outputs.append(output)

        final_arr = []

        for output in outputs:
            scores = np.array(output['scores'])
            output_labels = np.array(output['labels'])
            rearrange_map = [label_to_id[item] for item in output_labels]
            scores = scores[rearrange_map]
            output_labels = output_labels[rearrange_map]
            final_arr.append(scores)

        final_arr = torch.tensor(final_arr).to(device)

        return guid, final_arr,labels  

    def training_step(self, batch, batch_idx):
        raise Exception("Training step should not be called in zero shot scenario")
    

    def on_train_epoch_end(self):
        raise Exception("Training step should not be called in zero shot scenario")


    
    def validation_step(self, batch, batch_idx):
        raise Exception("Validation step should not be called in zero shot scenario")


    def test_step(self, batch, batch_idx):
        
        _guid, x,labels = self.forward(batch, batch_idx)
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
        #self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_accuracy', self.accuracy , on_step=False, on_epoch=True, prog_bar=True)
        return 0

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        _guid, x,_labels = self.forward(batch, batch_idx)
        preds = torch.argmax(x, dim=1)
        return preds


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer