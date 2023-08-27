import os
import json
import re

from utils import check_file_exists

FILE_PATH = '..\\dataset\\articles.json'
DEST_FILE_PATH = '..\\dataset\\complete_articles.json'

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

# Define a function to remove the three dots on each side of a match
def remove_dots(match):
    return match.group(1)

def replace_with_last(match):
    return match.group(1).split('|')[-1]


def remove_right_paranthesis_content_one_liners(text):

    regex = r"^\[\[[^\[\]]+\]\]$"
    
    new_text = re.sub(regex, '', text, flags=re.MULTILINE)

    if(new_text != text):
        return remove_right_paranthesis_content_one_liners(new_text)
    
    return text

def remove_right_paranthesis_content(text):

    regex = r"\[\[([^\[\]]+)\]\]"


    new_text = re.sub(regex, replace_with_last, text)

    if(new_text != text):
        return remove_right_paranthesis_content(new_text)
    
    return text

def remove_accolades_content(text):
    regex = r"\{\{([^\{\}]+)\}\}"
    new_text = re.sub(regex, '', text)

    if(new_text != text):
        return remove_accolades_content(new_text)
    
    return text

def remove_xml_tag(tag,text):
    regex = r"<" + tag + r"([^>]*)>(.*?)<\/" + tag + r">"

    new_text = re.sub(regex, r'\2', text, flags=re.DOTALL)

    if(new_text != text):
        return remove_xml_tag(tag, new_text)
    
    return new_text

def remove_html_comments(input_string):
    # The pattern for an HTML comment is <!-- followed by any number of
    # characters (non-greedy), and then -->
    pattern = re.compile("<!--.*?-->", re.DOTALL)
    no_comments_string = re.sub(pattern, '', input_string)
    return no_comments_string

def cleanup_pipeline(text, verbose=False):

    if(verbose):
        print('==================== ORIGINAL TEXT ======================')
        print(text)

    #return text
    text = remove_right_paranthesis_content_one_liners(text)
    # Remove {{multiple words}} and any content inside it
    text = remove_accolades_content(text)

    if(verbose):
        print('==================== REMOVED CONTENT ======================')
        print(text)


    # Remove paranthesis [[multiple words]] but keep content inside of it
    text = remove_right_paranthesis_content(text)


    text = re.sub(r"''wikt:([^|]+)\|[^|]+''", r'\1', text, flags=re.DOTALL)

    text = re.sub(r'{(.*?)}', ' ', text, flags=re.DOTALL)

    # Replace all substrings in the format '''substring that must be replaced''' with the substring without the three dots
    text = re.sub(r"'''(.*?)'''", remove_dots, text)


    # Remove all urls
    text = re.sub(r"\* \[http://[^\]]+\].*", '', text)
    text = re.sub(r"\* \[https://[^\]]+\].*", '', text)

    # Remove the external links
    text = re.sub(r"^Categorie:.*", '', text, flags=re.MULTILINE)

    # Remove files ( romanian format)
    text = re.sub(r"^Fișier:[^|]*", '', text, flags=re.MULTILINE)

    # Remove files ( international format)
    text = re.sub(r"^File:[^|]*", '', text, flags=re.MULTILINE)

    # Remove images ( local format)
    text = re.sub(r"^Imagine:[^|]*", '', text, flags=re.MULTILINE)

    # Remove images ( international format)
    text = re.sub(r"^Image:[^|]*", '', text, flags=re.MULTILINE)

    # Remove redirects
    text = re.sub(r"^\#REDIRECTEAZA.*", '', text, flags=re.MULTILINE)

    # Remove redirects
    text = re.sub(r"^\#REDIRECT.*", '', text, flags=re.MULTILINE)


    # Remove unwanted and unused referece tags
    text = text.replace("<references/>",'')
    text = text.replace("<references />",'')



    # regex that matches the content of the tag <noinclude>content</noinclude> and removes the content alongside with the tags. but it stops at the first closing tag
    text = re.sub(r'<noinclude>.*?</noinclude>', '', text, flags=re.DOTALL)

    # regex that matches the tag ref  with any possible attribute inside of it. it matches the content and stops at the first tag ending of </ref>
    text = re.sub(r"<ref(.*?)>(.*?)<\/ref>", '', text, flags=re.DOTALL)
    text = re.sub(r"<references(.*?)>(.*?)<\/references>", '', text, flags=re.DOTALL)

    text = re.sub(r"<gallery(.*?)>(.*?)<\/gallery>", '', text, flags=re.DOTALL)
    text = re.sub(r"<timeline(.*?)>(.*?)<\/timeline>", '', text, flags=re.DOTALL)
    text = re.sub(r"<div(.*?)>(.*?)<\/div>", '', text, flags=re.DOTALL)
    text = re.sub(r"<onlyinclude(.*?)>(.*?)<\/onlyinclude>", '', text, flags=re.DOTALL)
    #text = re.sub(r"<math(.*?)>(.*?)<\/math>", '', text, flags=re.DOTALL)

    # replace the content inside of tags
    text = remove_xml_tag('math', text)
    text = remove_xml_tag('small', text)

    # remove html comments
    text = remove_html_comments(text)

    # regex that matches the self closed ref tag with any possible attribute inside of it. it matches the content and stops at the first tag ending of />
    text = re.sub(r"<ref(.*?)\/[ ]*>", '', text, flags=re.DOTALL)

    # Remove simple tags
    text = text.replace("<br>", ' ')	

    # Remove empty double lines
    text = text.replace("\n\n", '\n')

    # Remove any self closed tags
    text = re.sub(r"<(.*?) />", '', text)

    text = re.sub(r"^\*[ ]$",'', text, flags=re.MULTILINE)

    text = re.sub(r"\bhttps?://\S+\b", '', text, flags=re.MULTILINE)

    # Remove unwanted whitespaces
    text = text.strip(" \n")


    if(text.startswith('#redirect')):
        return ''
    
    return text

def remove_empty_subcontent(items):
    remaining_items = [item for item in items if len(item["content"]) > 0]
    return remaining_items

def remove_small_subcontent(items, min_length):
    remaining_items = [item for item in items if len(item["content"]) > min_length]
    return remaining_items



def replace_and_remove(text, verbose=False):
    try:
        text = "==Descriere==\n" + text
    except:
        print(text)
        raise Exception("Error")

    regex = r"(==+)([^=\n]+?)\1\n"

    if(verbose):
        print(text)

    # split the text into subarticles

    # The list of the chapter titles
    splits = re.split(regex, text, flags=re.MULTILINE)
    splits.pop(0)
    

    matches = [splits[i] for i in range(len(splits)) if i%3 == 1]
    subarticles = [splits[i] for i in range(len(splits)) if i%3 == 2]

    # create a list of dictionaries
    result = []
    for i in range(0, len(matches)):
        chapter =  matches[i].strip()
        content =  subarticles[i]

        if(chapter in Banned_Chapters):
            continue
        result.append({"chapter": chapter, "content": content})

    # parse the content of each subarticle
    for subarticle in result:
        subarticle["content"] = cleanup_pipeline(subarticle["content"],verbose=verbose)

    if(verbose):
        raise Exception('asd')

    result = remove_small_subcontent(result, 50)
    #remove_empty_subcontent(result)

    return result


Banned_Chapters = [
    "Vezi \u0219i",
    "Leg\u0103turi externe",
    "Referin\u021be",
    "Vezi\u00a0\u0219i",
    "Imagini",
    "Leg\u0103turi\u00a0externe",
    "Lectur\u0103\u00a0suplimentar\u0103",
    "Bibliografie",
    "Note",
]

List = [
    'Ajutor',
    'Categorie',
    'Discuție',
    'Discuție Ajutor',
    'Discuție Categorie',
    'Discuție Fișier',
    'Discuție Format',
    'Discuție MediaWiki',
    'Discuție Portal',
    'Discuţie Portal',
    'Discuţie Proiect',
    'Discuție Utilizator',
    'Discuție Wikipedia',
    'Format',
    'Fișier',
    'MediaWiki',
    'Portal',
    'Proiect',
    'TimedText',
    'Modul',
    'Cod',
    'Utilizator',
    'Wikipedia',
]


def starts_with_any_of(string, start_templates):
    for template in start_templates:
        if(string.startswith(template + ':')):
            return True
    return False


def make_export(articles):
    export = []

    index = 0

    verbose = False


    for article in articles:
        if(index % 10000 == 0):
            print(index)
        index += 1

        #if(article['title'] == 'Mitologia greac\u0103'):
        #    print(article)
        #    verbose = True

        item = article['title'].split(':')[0]
        if(article['content'] is not None and not (starts_with_any_of(article['title'], List)) and  item.strip() not in List and 'Lista' not in article['title'] and 'Listă' not in article['title'] and len(article['content']) > 0):
            parsed_article = replace_and_remove(article['content'],verbose=verbose)
            parsed_article = {
                'title': article['title'],
                 'content': parsed_article
            }

            if(verbose):
                raise Exception('We have found the article')

            if(len(parsed_article['content']) > 0):
                export.append(parsed_article)
        elif(article['content'] is None and not(starts_with_any_of(article['title'], List))):
            print('we have found the following incompatiblity')
            print(article)
            print(starts_with_any_of(article['title'], List))
            print(article['title'].startswith('Utilizator:'))
            print('but why does it work so?')
            
    write_to_file(DEST_FILE_PATH, export)


content = read_json_from_file(FILE_PATH)
make_export(content)

