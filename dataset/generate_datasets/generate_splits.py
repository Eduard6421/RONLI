import json
import os
from collections import Counter
from sklearn.model_selection import train_test_split
import uuid

full_dataset = "nli_dataset.json"
test_dataset = "processed_items.json"


def create_train_set(lst):
    seen = set()
    for item in lst:
        item_tuple = (item['sentence1'],item['sentence2'],item['label'])
        seen.add(item_tuple)
    return seen


def create_test_set(lst):
    seen = set()
    for item in lst:
        item_tuple = (item['sentence1'],item['sentence2'],item['label'],item['corrected_label'])
        seen.add(item_tuple)
    return seen

def set_to_list(unique_set):
    unique_list = []
    for item in unique_set:
        unique_list.append({'sentence1': item[0],'sentence2': item[1],'label': item[2]})
        
    return unique_list

def set_to_list_test(unique_set):
    unique_list = []
    for item in unique_set:
        unique_list.append({'sentence1': item[0],'sentence2': item[1],'label': item[2],'corrected_label': item[3]})
    return unique_list


with open(full_dataset, "r",encoding='UTF-8') as all_data_file, open(test_dataset,"r",encoding="UTF-8") as test_data_file:
    train_dataset_full = json.load(all_data_file)
    test_dataset_full = json.load(test_data_file)

    train_set = create_train_set(train_dataset_full)
    test_set  = create_test_set(test_dataset_full)
    print(
        f"Number of training examples originally:"
    )
    print(len(train_dataset_full))

    print(
        f"Number of testing examples originally:"
    )
    print(len(test_dataset_full))


    list_train = set_to_list(train_set)


    train_data = []
    for item in train_set:
        if item not in test_set:
            train_data.append({'sentence1': item[0],'sentence2': item[1],'label': item[2]})


    test_data = set_to_list_test(test_set)

    print("Number of train examples afterwards:")
    print(len(train_data))
    label_counts = Counter(item["label"] for item in train_data)
    for label, count in label_counts.items():
        print(f"Label {label}: {count} occurrences")

    print("Number of test examples afterwards:")
    print(len(test_data))
    label_count_test = Counter(item["label"] for item in test_data)
    for label, count in label_count_test.items():
        print(f"Label {label}: {count} occurrences")

    # Split dataset intro train_validation
    train_split,validation_split = train_test_split(train_data,test_size=0.05,random_state=42, stratify=[item["label"] for item in train_data], shuffle=True)

    # Modify the label to be equal to corrected_label in the test set and remove corrected_label
    for item in test_data:
        item["label"] = item["corrected_label"]
        del item["corrected_label"]
        item['guid'] = str(uuid.uuid4())
    

    # Save the dataset

    with open("train.json", "w",encoding="utf-8") as train_file, open("validation.json", "w",encoding="utf-8") as validation_file, open("test.json", "w",encoding="utf-8") as test_file:
        json.dump(train_split, train_file, indent=4, ensure_ascii=False)
        json.dump(validation_split, validation_file, indent=4, ensure_ascii=False)
        json.dump(test_data, test_file, indent=4, ensure_ascii=False)
    