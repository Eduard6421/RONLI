# Read the processed_items.json file which is a utf-8 file

import json

from sklearn.metrics import cohen_kappa_score

data = None
with open('processed_items.json','r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# For each item in the processed_items take the label and corrected_label properties and try to compute the cohen kappa score

# get labels from first json
labels_orig = [ item['label'] for item in data ]

# get labels from second json
labels_adnotates = [ item['corrected_label'] for item in data ]

# compute kappa coefficient
kappa = cohen_kappa_score(labels_orig, labels_adnotates)

print(f'Kappa coefficient: {kappa}')