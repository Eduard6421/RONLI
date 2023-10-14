# Read json from validation.json
# and count number of items in each label


import json


with open("validation.json", "r", encoding="utf-8") as f:
    data = json.load(f)

labels = {}
for item in data:
    label = item["label"]
    if label not in labels:
        labels[label] = 1
    else:
        labels[label] += 1

print(labels)

print(labels[0] + labels[1] + labels[2] + labels[3])
