import os
import json
import re
import nltk
import random

from nltk.tokenize import sent_tokenize

from utils import check_file_exists


#nltk.download('punkt')

FILE_PATH = 'dataset\\complete_articles.json'
DEST_FILE_PATH = 'dataset\\nli_dataset.json'
NUM_UNRELATED_SENTENCES = 30000

"Function that reads json from file"
def read_json_from_file(filepath):
    if check_file_exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        print("File does not exist")

def write_to_file(filename, content):
    with open(filename, 'w',encoding='utf8') as f:
        json.dump(content, f, indent=4,ensure_ascii=False)

#Contrastante
contrast_stats = [
    'În contrast',
    'În sens opus',
    'În sens opus',
    'În opoziție',
    'În contrast',
    "În directă opoziție",
    'În contradicție',
    'În contradicție',
    'În contradicție însă',
    'Într-o opoziție',
    'Contradictoriu',
    'În contradictoriu',
    'Într-o opoziție radicală',
    'Într-o opoziție radicală',
    'Într-o opoziție clară',
    'Într-o opoziție clară',
    'Într-o opoziție evidentă',
    'Într-o opoziție evidentă',
    'Într-un dezacord total',  
    'Într-un dezacord total',
    'Cu toate acestea însă',
    'În ciuda acestui fapt',
    'În ciuda acestor fapte',
    'În ciuda acestui lucru',
    'În ciuda acestor lucruri',
    'În ciuda acestor motive',
    'În ciuda acestei certitudini',
    'În ciuda celor menționate',
    'În ciuda celor spuse',
    'În contradicție flagrantă',
    'Într-o opoziție clară',
    'Contrar impresiilor inițiale',
    'În mod diametral opus',
    'Într-un contrast',
    'Pe de cealaltă parte',
    'Pe de altă parte',
    'Într-o abordare opusă',
    'Contrar tuturor aparențelor',
    'Contrar convingerilor',
    'În antiteză',
    'În completă antiteză',
    'Totuși, contrar așteptărilor',
    'Contrastând',
    'Contrastând cu cele anterior menționate',
    'În contradictoriu',
    'În sens opus',
    'În mod antitetic',
    'În mod complet opus',
    'În directă opoziție',
    'În pofida acestor lucruri',
    'În pofida ascestui lucru',
    'Antagonic',
    'În mod antagonic',
    'Într-un antagonism',
    'În dezacord',
    'Contrar cu cele spuse',
    'Contrar cu cele spuse anterior',
    'În mod contrar',
    'Însă in mod contrar',
    ]
    #'Spre deosebire de',/
    #'Totuși, în cazul',

    #'Diferit de',
    #'Diferite de',
    #'Chiar dacă nu',
    #'Contrastând cu',
    #'Într-o notă diferită',
    #'În schimb',
    #exprimă o idee contrară
entailment_stats = [
#'Atunci însă',
'Cu alte cuvinte',
'Altfel spus',
'În alți termeni',
'Pus altfel',
'Asta înseamnă că',
'Adică',
'În traducere liberă',
'În termeni mai clari',
'Sau, mai bine zis',
'Într-o formulare diferită',
'Simplu spus',
'Rezumând',
'Simplificând',
'În rezumat',
'În termeni simpli',
'În traducere liberă',
'Mai concis',
'Într-o formulare mai concisă',
'În esență',
'Într-o formulare mai simplă',
'Sinteza este că',
'Sintetizând',
'Mai pe scurt',
'Într-o formulare mai scurtă',
'Pe larg',
'Într-o formulare mai lungă',
'Într-o formulare mai clară',
'Mai pe șleau',
'În termeni mai puțin academici',
'În termeni populari',
'Într-o altă formulare',
'În fond',
]

rational_stats = [
'Astfel că',
'Prin urmare',
'În consecință',
'Ca urmare',
'Ca rezultat',
'Concluzionând',
'Provocând astfel',
'Ceea ce a dus la',
'Ceea ce duce la',
'Ceea ce provoacă',
'Rezultatul este',
'Rezultând din',
'Conducând la',
'Așadar',
'Într-o concluzie',
'Aducând la',
'Ducând la',
'Pentru a finaliza',
'Aceasta duce la',
'În rezultat',
'În acest fel',
'Ceea ce cauzează',
'Ceea ce conduce la',
'Astfel, rezultă că',
'Prin urmare',
'Ca o consecință a acestui fapt',
'În concluzie',
'Din această cauză',
'Într-o concluzie',
'Din aceasta cauză',
'Se poate concluziona că',
'În sumă',
'Ținând cont de acestea',
'Rezultând din acestea',
'Rezultând că',
'Drept urmare',
'Astfel'
]



def append_symbol_to_items(strings, symbol):
    modified_strings = [string + symbol for string in strings]
    return modified_strings


def extract_sentence_pairs(text_sentences, predefined_expressions):


    #extended_expressions_dots = append_symbol_to_items(predefined_expressions, ':')
    #extended_expressions_comma = append_symbol_to_items(predefined_expressions, ',')

    #predefined_expressions += extended_expressions_dots
    #predefined_expressions += extended_expressions_comma

    # Tokenize the text into sentences
    sentence_pairs = []
    # Group the sentences into pairs, and add to list if match is found
    for i in range(len(text_sentences) - 1):
        if any(express in text_sentences[i+1] for express in predefined_expressions):

            second_string_processed = remove_templates(text_sentences[i+1], predefined_expressions)
            if(len(second_string_processed) > 0):
                sentence_pairs.append((text_sentences[i], second_string_processed))

    return sentence_pairs


def generate_random_pairs(n_pairs, lower_bound, upper_bound):
    pairs = set()

    while(len(pairs) < n_pairs):
        num1 = random.randint(lower_bound, upper_bound)
        num2 = random.randint(lower_bound, upper_bound)
        
        if num1 < num2:
            pairs.add((num1, num2))
        else:
            pairs.add((num2, num1))

    return list(pairs)


def remove_templates(input_string, expressions):
    # Check if any expression from the list matches the beginning of the string
    for expr in expressions:
        if input_string.startswith(expr):
            # Remove the matched part from the string
            input_string = input_string[len(expr):]
            break

    # Remove all content before the first letter (punctuation signs etc.)
    input_string = re.sub(r'^\W*', '', input_string)

    # Capitalize the first letter
    input_string = input_string.capitalize()

    return input_string


def parse_nli(articles, output_file):

    dataset = []

    all_sentences = []


    num_label_0 = 0
    num_label_1 = 0
    num_label_2 = 0
    num_label_3 = 0
    
    for idx, article in enumerate(articles):
        title = article['title']
        contents = article['content']

        if(idx%10000 == 0):
            print('Parsing article {}'.format(idx))

        for entry in contents:
            text = entry['content']
            #print('Parsing article {} - chapter {}'.format(idx, entry['chapter']))
            chapter_name = entry['chapter']

            text_sentences = sent_tokenize(text)

            contrastive_sentence_pairs = extract_sentence_pairs(text_sentences, contrast_stats)
            entailment_sentence_pairs = extract_sentence_pairs(text_sentences, entailment_stats)
            rational_sentence_pairs = extract_sentence_pairs(text_sentences, rational_stats)

            for dataset_item in contrastive_sentence_pairs:

                dataset.append({
                    'sentence1': dataset_item[0],
                    'sentence2': dataset_item[1],
                    'label': 0
                })
                num_label_0 +=1 
            for dataset_item in entailment_sentence_pairs:
                dataset.append({
                    'sentence1': dataset_item[0],
                    'sentence2': dataset_item[1],
                    'label': 1
                })
                num_label_1 +=1
            for dataset_item in rational_sentence_pairs:
                dataset.append({
                    'sentence1': dataset_item[0],
                    'sentence2': dataset_item[1],
                    'label': 2
                })
                num_label_2 +=1

            all_sentences += [filtered_sentence for filtered_sentence in text_sentences if len(filtered_sentence) > 50]

    num_all_items = len(all_sentences)


    pairs = generate_random_pairs(NUM_UNRELATED_SENTENCES, 0, num_all_items)

    for pair in pairs:
        dataset.append({
            'sentence1': all_sentences[pair[0]],
            'sentence2': all_sentences[pair[1]],
            'label': 3
        })
        num_label_3 +=1
    
    print("Number of label 0: {}".format(num_label_0))
    print("Number of label 1: {}".format(num_label_1))
    print("Number of label 2: {}".format(num_label_2))
    print("Number of label 3: {}".format(num_label_3))
    
    write_to_file(output_file, dataset)

articles = read_json_from_file(FILE_PATH)
print('ended read')
parse_nli(articles=articles, output_file = DEST_FILE_PATH)