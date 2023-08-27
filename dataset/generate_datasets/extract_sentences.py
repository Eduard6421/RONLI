
import json 
def load_data( data_path):
        with open(data_path, "r",encoding="utf-8") as f:
            data = json.load(f)
        return data

dataset = load_data("./dataset/nli_dataset copy.json")


# write a function that goes throgh all items in dataset
# and extract the property "label" and marks in a dictonary how many times each label appears

def count_labels(dataset):
    label_count = {}
    for item in dataset:
        label = item["label"]
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    return label_count


print(count_labels(dataset))