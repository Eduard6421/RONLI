# write a program that reads train_easy.json and train_ambiguous.json
# and reads the arrays inside of them
# and then checks how many guids they have in common

import json

with open("train_easy.json", "r", encoding="utf-8") as f:
    train_easy = json.load(f)

with open("train_ambiguous.json", "r", encoding="utf-8") as f:
    train_ambiguous = json.load(f)


# create another json file with the cominbations of the two
# while removing any common guid

new_set = []

new_guids = []
for item in train_easy:
    new_set.append(item)
    new_guids.append(item["guid"])
for item in train_ambiguous:
    if item["guid"] not in new_guids:
        new_set.append(item)

with open("train_easy_ambiguous.json", "w", encoding="utf-8") as f:
    json.dump(new_set, f, indent=4)
