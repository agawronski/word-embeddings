#-------------------------------------------------------------------------------
# pip3 install install beautifulsoup4
# pip3 install sentence-transformers
# pip3 install pandas
from urllib import request
from io import StringIO
import json
import os
import re

from sentence_transformers import SentenceTransformer
from nltk.probability import FreqDist
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import string
import nltk

print('starting')

nltk.download('stopwords')
nltk.download('punkt')

sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")

print('nltk and transformer downloaded')

def url_2_text(article_url):
    html = request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.findAll(text=True)
    output = ''
    blacklist = ['[document]','noscript','header','html','meta','head','input','script']
    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)
    return output


def strip_punctuation_stopwords(text):
    return [word.strip(string.punctuation).lower() for word in text.split() if word not in nltk.corpus.stopwords.words()]


def get_average_embedding(token_list):
    embeddings = sentence_model.encode(token_list, show_progress_bar=False)
    average_embeding = embeddings.mean(axis=0)
    return average_embeding


def make_dicts_and_lists_strings(one_article_dict):
    for key in one_article_dict.keys():
        if isinstance(one_article_dict[key], list) or isinstance(one_article_dict[key], dict):
            one_article_dict[key] = str(one_article_dict[key])


print('functions defined')

print('starting to load data')

with open('healthcare_industry_from_1900_2021.jsonl') as f:
    lines = f.readlines()


print('read lines done')
print(len(lines))

outlist = []
for record in lines:
    one_record_dict = json.loads(record)
    make_dicts_and_lists_strings(one_record_dict)
    outlist.append(one_record_dict)


data = pd.DataFrame(outlist)


data.loc[data.shape[0]] = [None]*data.shape[1]

url = 'https://www.neowin.net/news/canon-slapped-with-lawsuit-for-disabling-scans-when-printers-are-out-of-ink/'
output = url_2_text(url)
output = strip_punctuation_stopwords(output)

data.loc[data.shape[0]-1,'fullText'] = ' '.join(output)

print(data.shape)

print('starting for loop')

outlist = []
for index, row in data.iterrows():
    print(index)
    if index % 20 == 0:
        print('saving')
        outdf = pd.DataFrame(outlist)
        outdf.to_csv('20211017_embeddings_saved.csv')
    clean_text = strip_punctuation_stopwords(row['fullText'])
    avg_embedding = get_average_embedding(list(set(clean_text)))
    outlist.append(avg_embedding)


outdf = pd.DataFrame(outlist)

outdf.to_csv('20211017_embeddings_saved_FULL.csv')
