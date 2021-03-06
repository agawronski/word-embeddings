from urllib import request
from io import StringIO
import json
import sys
import os
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer, util
from nltk.probability import FreqDist
from nltk.stem import LancasterStemmer
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import spacy_streamlit
import streamlit as st
import pandas as pd
import numpy as np
import spacy
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# https://www.sbert.net/docs/pretrained_models.html
sentence_model = SentenceTransformer("all-mpnet-base-v2")


st.title('GLG project')
spacy_model = "en_core_web_sm"
DEFAULT_TEXT='complete text'
text = st.text_area("Text to analyze", DEFAULT_TEXT, height=200)
doc = spacy_streamlit.process_text(spacy_model, text)

spacy_streamlit.visualize_ner(
    doc,
    labels=nlp.get_pipe("ner").labels,
    title="Named Entities",
    show_table=False
)


@st.cache
def load_data(file):
    data = pd.read_csv(file, index_col=0)
    return data


data = pd.read_csv('https://word-emeddings.s3.us-west-2.amazonaws.com/20211102_WIKI_1_weighted_embeddings_saved_FULL.csv')
article_df = pd.read_csv('https://word-emeddings.s3.us-west-2.amazonaws.com/people_wiki.csv')


data = data.sample(n=2000, random_state=111)
article_df = article_df.loc[data.index,:]

data = data.reset_index(drop=True)
article_df = article_df.reset_index(drop=True)

print(data.shape)
print(article_df.shape)

def strip_punctuation_stopwords(token_list):
    return [word.lower() for word in token_list \
                if word.lower() not in nltk.corpus.stopwords.words() \
                and word not in string.punctuation \
                and len(word) > 1]

st.subheader('Documents')
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
    final = pd.DataFrame(weighted_embed.sum(axis=0))
    return final

weighted_embedding = get_weighted_embedding(text)
weighted_embedding2 = weighted_embedding.T
weighted_embedding2.columns = data.columns
print(data.head())
print(data.tail())
print(weighted_embedding2)
print(weighted_embedding2.columns, file=sys.stderr)
data2 = pd.concat([data, weighted_embedding2]).reset_index(drop=True)
print(data2.shape)
print(data2.head())
print(data2.tail())
print('DID IT MAKE SENSE?')
print(data2.dtypes)
dist_mat = distance_matrix(data2, data2)
dist_mat = pd.DataFrame(dist_mat)
print('dist_mat - tail')
print(dist_mat.tail(10))
print(dist_mat.tail(1).T)
data2['dist_2_new'] = dist_mat.tail(1).T
print('data2 head BEFORE SORT')
print(data2.head())
print('data2 tail BEFORE SORT')
print(data2.tail())
data2 = data2.sort_values('dist_2_new')
print('data2 head:')
print(data2.head())
print('data2 tail after sort:')
print(data2.tail())
article_index = [x for x in data2.head().index if x != 2000]
dataF = article_df.loc[article_index,:].copy()
# dataF = dataF.loc[:,['abstract', 'creator', 'datePublished']]
st.dataframe(dataF)
