#-------------------------------------------------------------------------------
# ssh -i /Users/aidangawronski/Documents/fourth_brain_capstone/fourth-brain-basic-ec2.pem ec2-user@ec2-52-13-29-251.us-west-2.compute.amazonaws.com
# nohup python3 -u get_embeddings_V3.py > output_V3.log &
# pip3 install install beautifulsoup4
# pip3 install sentence-transformers
# pip3 install pandas
from urllib import request
from io import StringIO
import json
import os
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from nltk.probability import FreqDist
from nltk.stem import LancasterStemmer
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt')

sentence_model = SentenceTransformer("all-mpnet-base-v2")


def strip_punctuation_stopwords(token_list):
    return [word.lower() for word in token_list \
                if word.lower() not in nltk.corpus.stopwords.words() \
                and word not in string.punctuation \
                and len(word) > 1]


def get_average_embedding(token_list):
    embeddings = sentence_model.encode(token_list, show_progress_bar=False)
    average_embeding = embeddings.mean(axis=0)
    return average_embeding


def get_weighted_embedding(token_list_long):
    text_token = word_tokenize(token_list_long)
    text_token_clean = strip_punctuation_stopwords(text_token)
    token_counts = FreqDist(text_token_clean)
    token_counts = pd.DataFrame(token_counts, index=[0]).T
    token_counts.columns = ['counts']
    token_counts = token_counts.sort_values('counts', ascending=False).head(30)
    token_counts['weight'] = token_counts/token_counts.sum()
    embeddings = sentence_model.encode(token_counts.index, show_progress_bar=False)
    weighted_embed = np.dot(np.diag(token_counts['weight']), embeddings)
    final = weighted_embed.sum(axis=0)
    return final


data = pd.read_csv('https://word-emeddings.s3.us-west-2.amazonaws.com/people_wiki.csv')

outlist = []
for index, row in data.iterrows():
    if index % 10 == 0:
        print(index)
    if index % 200 == 0:
        print(row['text'])
        print('saving')
        outdf = pd.DataFrame(outlist)
        outdf.to_csv('20211102_WIKI_1_weighted_embeddings_saved.csv', index=False)
    weighted_embedding = get_weighted_embedding(row['text'])
    outlist.append(weighted_embedding)

outdf = pd.DataFrame(outlist)

outdf.to_csv('20211102_WIKI_1_weighted_embeddings_saved_FULL.csv', index=False)











#-------------------------------------------------------------------------------

