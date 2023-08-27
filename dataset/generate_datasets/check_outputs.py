import json 
# open the text file and read all lines:

word_to_class ={
    'Contrastive': 0,
    'Entailment': 1,
    'Consequence': 2,
    'Unrelated': 3
}

dict = {
    "Contrastive": 0,
    "Entailment": 0,
    "Consequence": 0,
    "Unrelated": 0
}
preds = []
with open("temp_preds.txt", "r", encoding="utf-8") as f:
    outputs = json.load(f)
    # count how many items item appears
    for idx,item in enumerate(outputs):
        if item.split()[-1] in dict:
            preds.append(word_to_class[item.split()[-1]])
            dict[item.split()[-1]] += 1
        else:
            raise Exception("wtf")
        #if(idx < 1000):
        #    print(idx, len(preds))

gt_labels = []
with open("dataset/datasets/test.json", encoding="UTF-8") as f:
    gt_items = json.load(f)
    gt_labels = [item['label'] for item in gt_items]

from sklearn.metrics import classification_report

#print(gt_labels)
#print(preds)

print(gt_labels)
print(preds)
print(classification_report(gt_labels, preds, digits=4))