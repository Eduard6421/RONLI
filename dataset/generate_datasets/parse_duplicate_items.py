import json

data_orig = None
with open('dataset/datasets/train_easyambiguous.json','r', encoding='utf-8') as json_file:
    data_orig = json.load(json_file)

data_adnotates = None
with open('dataset/datasets/train_hard.json','r', encoding='utf-8') as json_file:
    data_adnotates = json.load(json_file)


# get labels from first json
labels_orig = [ item['label'] for item in data_orig ]

# get labels from second json
labels_adnotates = [ item['label'] for item in data_adnotates ]

guid_set = set([])
final_train_items = []

# Add items from the two json to the train items as long as the guid is not already in the train items
for item in data_orig:
    if item['guid'] not in guid_set:
        final_train_items.append(item)
        guid_set.add(item['guid'])

for item in data_adnotates:
    if item['guid'] not in guid_set:
        final_train_items.append(item)
        guid_set.add(item['guid'])

# write it to a file
with open('dataset/datasets/train_curriculum_full.json', 'w', encoding='utf-8') as f:
    json.dump(final_train_items, f, ensure_ascii=False, indent=4)