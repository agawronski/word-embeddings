#-------------------------------------------------------------------------------
# ssh -i /Users/aidangawronski/Documents/fourth_brain_capstone/fourth-brain-basic-ec2.pem ec2-user@ec2-52-13-29-251.us-west-2.compute.amazonaws.com
# cd fourth_brain_mvp_1/
from urllib import request
from io import StringIO
import json
import os
import re

from sentence_transformers import SentenceTransformer
from nltk.probability import FreqDist
from sklearn.cluster import KMeans
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt')

sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")

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


with open('healthcare_industry_from_1900_2021.jsonl') as f:
    lines = f.readlines()


outlist = []
for record in lines:
    one_record_dict = json.loads(record)
    make_dicts_and_lists_strings(one_record_dict)
    outlist.append(one_record_dict)


data = pd.DataFrame(outlist)

data.shape
# (1500, 34)

data.loc[data.shape[0]] = [None]*data.shape[1]

url = 'https://www.neowin.net/news/canon-slapped-with-lawsuit-for-disabling-scans-when-printers-are-out-of-ink/'
output = url_2_text(url)
output = strip_punctuation_stopwords(output)

data.loc[data.shape[0]-1,'fullText'] = ' '.join(output)

data.shape
# (1501, 34)

# data.to_csv('20211024_main_article_dataframe.csv', index=False)

embeddings = pd.read_csv('20211017_embeddings_saved_FULL.csv', index_col=0)

embeddings.shape
# (1501, 768)

from scipy.spatial import distance_matrix

dist_mat = distance_matrix(embeddings, embeddings)
dist_mat = pd.DataFrame(dist_mat)

data['dist_2_new'] = dist_mat.tail(1).T

# fullText
kmeans = KMeans(n_clusters = 20, random_state = 1111)
clusters = kmeans.fit_predict(embeddings)

data['clusters'] = clusters

data.clusters.value_counts()

data2 = data.sort_values('dist_2_new')

for index, row in data2.head(10).iterrows():
    print('----------------')
    print(row.abstract)
    print(f'Article Cluster --> {row.clusters}')





























#-------------------------------------------------------------------------------
