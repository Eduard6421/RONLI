import json
import uuid


def add_guid_to_objects(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for obj in data:
        obj['guid'] = str(uuid.uuid4())
    
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)



add_guid_to_objects('train.json')
add_guid_to_objects('validation.json')
add_guid_to_objects('test.json')