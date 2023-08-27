import json

# Load the contents of the first JSON file
with open('validation.json', 'r', encoding='utf-8') as f1:
    data1 = json.load(f1)

# Load the contents of the second JSON file
with open('processed_items.json', 'r', encoding='utf-8') as f2:
    data2 = json.load(f2)

# Create a dictionary for faster lookups based on sentence1 and sentence2 from the first file
guid_lookup = {(item['sentence1'], item['sentence2']): item['guid'] for item in data1}

# Iterate over the items in the second file and add the guid from the first file
for item in data2:
    key = (item['sentence1'], item['sentence2'])
    if key in guid_lookup:
        item['guid'] = guid_lookup[key]

# Save the updated second file with the added guids
with open('processed_items_updated.json', 'w', encoding='utf-8') as f:
    json.dump(data2, f, indent=4,  ensure_ascii=False,)