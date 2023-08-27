import json
import csv

# Read the json file
with open('dataset/datasets/train.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Prepare the data for tsv
data_to_write = []
for item in data:
    data_to_write.append([item['guid'], item['sentence1'], item['sentence2'], item['label']])

# Write the tsv file
with open('train.tsv', 'w', newline='', encoding='utf-8') as tsv_file:
    writer = csv.writer(tsv_file, delimiter='\t')
    writer.writerow(['guid', 'sentence1', 'sentence2', 'label'])  # Write the header
    writer.writerows(data_to_write)  # Write the data rows
