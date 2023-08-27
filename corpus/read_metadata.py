import os
import json
import re

from utils import check_file_exists


FILE_PATH = '..\\dataset\\articles.json'
DEST_FILE_PATH = '..\\dataset\\category_tags.json'

"Function that reads json from file"
def read_json_from_file(filepath):
    if check_file_exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        print("File does not exist")

def write_to_file(filename, content):
    with open(filename, 'w') as f:
        json.dump(content, f, indent=4)

def read_file(file_name):
    if(not check_file_exists(file_name)):
        print("File does not exist")
        return
    
    with open(file_name, 'r') as f:
        return f.read()
    


def parse_metadata(articles, destination_file):

    category_tags = set([])

    article_tags = []

    for idx, item in enumerate(articles):

        if(idx % 10000 == 0):
            print("Parsed {} articles".format(idx))

        title = item['title']
        content = item['content']

        pattern = r"^\[\[Categorie:([^\[:\]]+)\]\]$"

        identified_tags = [] 

        if(content is not None):
            identified_tags = re.findall(pattern, content, flags=re.MULTILINE)

        article_tags.append({'title': title, 'tags':identified_tags})
        category_tags.update(identified_tags)
        #category_tags.extend(identified_tags)

    write_to_file(destination_file, sorted(list(category_tags)))


articles = read_json_from_file(FILE_PATH)
parse_metadata(articles=articles, destination_file = DEST_FILE_PATH)


