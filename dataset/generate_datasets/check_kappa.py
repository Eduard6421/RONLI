from sklearn.metrics import cohen_kappa_score

# Compute the kappa coefficient for two annotators
# Read the file sampled_test.json that is utf-8 in format. it contains a json
import json

data_orig = None
with open('sampled_test.json','r', encoding='utf-8') as json_file:
    data_orig = json.load(json_file)

data_adnotates = None
with open('sampled_adnotates.json','r', encoding='utf-8') as json_file:
    data_adnotates = json.load(json_file)


# For each guid in the sampled_test.json, find the corresponding guid in the sampled_adnotates.json
# and get the label from there. Then, compare the label from sampled_test.json with the label from
# sampled_adnotates.json and compute the kappa coefficient for the two annotators

# get labels from first json
labels_orig = [ item['label'] for item in data_orig ]

# get labels from second json
labels_adnotates = [ item['label'] for item in data_adnotates ]


# compute kappa coefficient
kappa = cohen_kappa_score(labels_orig, labels_adnotates)
print(f'Kappa coefficient: {kappa}')