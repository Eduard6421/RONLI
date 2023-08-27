from collections import defaultdict
import json
import random
import os

# Function to clear the terminal
def clear_terminal():
    if os.name == 'posix':  # For Unix/Linux/MacOS
        os.system('clear')
    elif os.name == 'nt':  # For Windows
        os.system('cls')

# Function to select random items with stratified sampling
def select_random_items(data, existing_data, num_items, ):
    label_counts = {}
    preloaded_label_counts = {}
    selected_items = []


    for item in existing_data:
        label = item['label']
        preloaded_label_counts[label] = preloaded_label_counts.get(label, 0) + 1
    
    # Count the occurrences of each label
    for item in data:
        label = item['label']
        label_counts[label] = label_counts.get(label, 0) + 1


    print(label_counts)
    print(label_counts[0] + label_counts[1]  + label_counts[2]  + label_counts[3])

    
    # Calculate the proportion of items to select per label based on the desired number of items
    label_proportions = {label: count / len(data) for label, count in label_counts.items()}
    num_items_per_label = {label: round(label_proportions[label] * num_items) for label in label_proportions}

    for key in num_items_per_label:
        if key in preloaded_label_counts:
            num_items_per_label[key] = num_items_per_label[key] - preloaded_label_counts[key]

    filtered_data = []
    for item in data:
        skip = False
        for selected_item in existing_data:
            if item['sentence1'] == selected_item['sentence1'] and item['sentence2'] == selected_item['sentence2']:
                skip = True
        if skip == False:
            filtered_data.append(item)

    print(num_items_per_label)
    print(preloaded_label_counts)

    # Select items proportionally from each label category
    for label, count in num_items_per_label.items():
        items_with_label = [item for item in filtered_data if item['label'] == label]

        print(count)

        if(count > 0):
            selected_items.extend(random.sample(items_with_label, count))


    # sort selecvted items randomly
    random.shuffle(selected_items)
    
    return selected_items

# Function to interact with the user and collect input
def collect_user_input(item):
    print("Please select the relationship:")
    print()
    print(item['sentence1'])
    print()
    print(item['sentence2'])
    print()
    print("0 = contrastive, 1 = entailment, 2 = consequence, 3 = unrelated")
    user_input = input("Enter a number between 0 and 3: ")

    if(int(user_input) not in [0,1,2,3]):
        raise Exception('Unkown input')
    return int(user_input)

output_file = 'processed_items.json'
existing_data = []
if os.path.exists(output_file):
    with open(output_file, 'r',encoding='utf-8') as file:
        existing_data = json.load(file)

# Load the JSON data from memory
with open('validation.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

    # Select 3000 random items with proportional labels
    selected_items = select_random_items(json_data,existing_data, 3059)

    # Process the selected items
    processed_items = existing_data

    for item in selected_items:
        clear_terminal()
        user_input = collect_user_input(item)
        corrected_item = {
            'sentence1': item['sentence1'],
            'sentence2': item['sentence2'],
            'label': item['label'],
            'corrected_label': user_input,
            'guid' : item['guid']
        }
        processed_items.append(corrected_item)

        # Save the processed items to a local file
        with open(output_file, 'w',encoding='utf-8') as outfile:
            json.dump(processed_items, outfile, ensure_ascii=False, indent=4)

    print("Processing completed and data saved to 'processed_items.json'.")