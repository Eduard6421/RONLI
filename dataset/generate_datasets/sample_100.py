import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data from JSON file
with open('train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract labels
labels = [d['label'] for d in data]

# Stratified sampling with train_test_split
_, sample_indices = train_test_split(list(range(len(data))), test_size=100, stratify=labels, random_state=42)

# Select the sampled items
sampled_data = [data[i] for i in sample_indices]

# Save the sampled data into a JSON file
with open('sampled_test.json', 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, indent=4, ensure_ascii=False)