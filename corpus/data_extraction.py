from lxml import etree

import json
import os

from cleanup.utils import check_file_exists

XML_FILEPATH = '..\\dataset\\wiki.xml'
ARTICLES_FILEPATH = '..\\articles.json'




# function that parses a large xml file and prints out the top level tags
def parse_xml_file(filepath):
    if check_file_exists(filepath):
        context = etree.iterparse(filepath, events=('start', 'end'))
        for event, elem in context:
            if event == 'start':
                print(elem.tag)
            elem.clear()
    else:
        print("File does not exist")



# function that recursively prints the xml subtags of a given xml tag by using lxml
# try to print it in a more readable way by keeping track of the depth of the tag
def print_subtags(tag, depth=0):
    print('  ' * depth + tag.tag.removeprefix('{http://www.mediawiki.org/xml/export-0.10/}')) 
    #if(tag.text):
    #    print('  ' * depth + tag.text)
    for child in tag:
        print_subtags(child, depth + 1)



# function that parses a large xml file and reads all the tags with a given tag name
# for each tag identified it searches for another two subtags named text, title and saves the text in an object
def parse_xml_contents(filepath):
    if check_file_exists(filepath):
        wikia_content = []

        context = etree.iterparse(filepath, events=('end',), tag='{http://www.mediawiki.org/xml/export-0.10/}page')
        # add a filter if the tag is not in the list of tags

        # add a counter to each article and print it
        counter = 0

        for event, elem in context:
            title = elem.find('{http://www.mediawiki.org/xml/export-0.10/}title').text
            try:
                content = elem.find('{http://www.mediawiki.org/xml/export-0.10/}revision/{http://www.mediawiki.org/xml/export-0.10/}text').text
            except:
                print(elem)
                print(elem.tag)
                for event,elem in context.children:
                    print(f"{event} - {elem.tag}")
                raise  Exception("No content found")
            wikia_article = {
                'title': title,
                'content': content
            }
            wikia_content.append(wikia_article)

            counter +=1
            if(counter % 1000 == 0):
                print(counter)
            #print('the content of the article is:')
            #print(wikia_content)
            #break
        
        return wikia_content

    else:
        print("File does not exist")


# write the articles to a file
def write_articles_to_file(filepath, articles):
    if not os.path.isfile(filepath):
        with open(filepath, 'w') as f:
            json.dump(articles, f, indent=4)
    else:
        print("File already exists")

# function that parses al arge xml file and prints the content of the tag with the given name
def parse_xml_file_tags(filepath, tags):
    if check_file_exists(filepath):
        context = etree.iterparse(filepath, events=('start', 'end'))
        # add a filter if the tag is not in the list of tags
        for event, elem in context:        
            if event == 'start' and elem.tag in tags:
                print('=================================== Article Start ===================================')
                print_subtags(elem)
            elif(event == 'end' and elem.tag in tags):
                print('=================================== Article End ===================================')                
            elem.clear()
    else:
        print("File does not exist")


def main():
    articles = parse_xml_contents(XML_FILEPATH)
    write_articles_to_file(ARTICLES_FILEPATH, articles)

main()