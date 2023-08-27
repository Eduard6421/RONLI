from collections import Counter
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,cohen_kappa_score
processed_data  = None


# Loading confusion matrxi, accuracy score and cohen kappa for train and test

with open("processed_items.json","r",encoding="utf-8") as processed_file:
    processed_data = json.load(processed_file)

original_labels = [ item['label'] for item in processed_data]
corrected_labels = [item['corrected_label'] for item in processed_data]

print("Confusion matrix ")
print(confusion_matrix(corrected_labels,original_labels))
print(f"Cohen Kappa: {cohen_kappa_score(corrected_labels,original_labels)}")


print()
print()

# Read train,test,validation dataset

train_data = None
validation_data = None
test_data = None

with open("datasets/train.json","r",encoding="utf-8") as train_file, open("datasets/validation.json","r",encoding="utf-8") as validation_file, open("datasets/test.json","r",encoding="utf-8") as test_file:
    train_data = json.load(train_file)
    validation_data = json.load(validation_file)
    test_data = json.load(test_file)


def make_dataset_statistics(data, data_name):
    # Print number of labels for train
    print(f"Number of {data_name} examples: {len(data)}")
    print(f"Label distribution for {data_name}:")
    label_counts = Counter(item["label"] for item in data)
    for label, count in label_counts.items():
        print(f"Label {label}: {count} occurrences")
    # Compte average length of sentence1 in train
    average_num_sentence_1_words = sum(len(item["sentence1"].split()) for item in data) / len(data)
    average_num_sentence_2_words = sum(len(item["sentence2"].split()) for item in data) / len(data)


    print(f"Average length of sentence1 in {data_name}: {average_num_sentence_1_words}")
    print(f"Average length of sentence2 in {data_name}: {average_num_sentence_2_words}")
    # Print average length of inputs
    # concatenate sentence 1 and sentence 2 and compute average length
    average_num_words = average_num_sentence_2_words + average_num_sentence_1_words #sum(len(item["sentence1"].split()) + len(item["sentence2"].split()) for item in data) / len(data)
    print(f"Average length of inputs in {data_name}: {average_num_words}")
    # For each label compute the average number of common words between sentence1 and sentence2
    #for label in label_counts.keys():
    #    average_num_common_words = sum(len(set(item["sentence1"].split()) & set(item["sentence2"].split())) for item in data if item["label"] == label) / label_counts[label]
    #    print(f"Average number of common words for label {label}: {average_num_common_words}")

    # Compute average percentage of overlap between sentence1 and sentence2
    percentages = []
    for item in data:
        sentence1 = item['sentence1']
        sentence2 = item['sentence2']
        label = item['label']

        # Compute percentage of overlap between sentence1 and sentence2
        percentage_overlap = len(set(sentence1.split()) & set(sentence2.split())) / len(set(sentence1.split()) | set(sentence2.split()))
        percentages.append(percentage_overlap)
    
    print(f"Average percentage of overlap between sentence1 and sentence2 in {data_name}: {sum(percentages) / len(percentages)}")

    print()
    # Compute average percentage of common words between sentence1 and sentence2


    # For each label compute average sentence1 length
    for label in label_counts.keys():
        average_num_sentence_1_words = sum(len(item["sentence1"].split()) for item in data if item["label"] == label) / label_counts[label]
        print(f"Average length of sentence1 for label {label} in {data_name}: {average_num_sentence_1_words}")
        # For each label compute average sentence2 length
        average_num_sentence_2_words = sum(len(item["sentence2"].split()) for item in data if item["label"] == label) / label_counts[label]
        print(f"Average length of sentence2 for label {label} in {data_name}: {average_num_sentence_2_words}")
        # For each label compute average input length
        average_num_words = average_num_sentence_2_words + average_num_sentence_1_words #sum(len(item["sentence1"].split()) + len(item["sentence2"].split()) for item in data if item["label"] == label) / label_counts[label]
        print(f"Average length of inputs for label {label} in {data_name}: {average_num_words}")
        
        # Compute average percentage of overlap between sentence1 and sentence2
        percentages = []
        for item in data:
            sentence1 = item['sentence1']
            sentence2 = item['sentence2']
            sub_label = item['label']

            if sub_label == label:
                # Compute percentage of overlap between sentence1 and sentence2
                percentage_overlap = len(set(sentence1.split()) & set(sentence2.split())) / len(set(sentence1.split()) | set(sentence2.split()))
                percentages.append(percentage_overlap)

        print(f"Average percentage of overlap between sentence1 and sentence2 for label {label} in {data_name}: {sum(percentages) / len(percentages)}")
        print()

    # Average sentence1 length
    average_num_sentence_1_words = sum(len(item["sentence1"].split()) for item in data) / len(data)
    print(f"Average length of sentence1 in {data_name}: {average_num_sentence_1_words}")

    # Average sentence2 length
    average_num_sentence_2_words = sum(len(item["sentence2"].split()) for item in data) / len(data)
    print(f"Average length of sentence2 in {data_name}: {average_num_sentence_2_words}")

    # Average overlap between sentence1 and sentence2
    percentages = []
    for item in data:
        sentence1 = item['sentence1']
        sentence2 = item['sentence2']
        # Compute percentage of overlap between sentence1 and sentence2
        percentage_overlap = len(set(sentence1.split()) & set(sentence2.split())) / len(set(sentence1.split()) | set(sentence2.split()))
        percentages.append(percentage_overlap)
    
    print(f"Average percentage of overlap between sentence1 and sentence2 in {data_name}: {sum(percentages) / len(percentages)}")


make_dataset_statistics(train_data,"train")
make_dataset_statistics(validation_data,"validation")
make_dataset_statistics(test_data,"test")




