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

nltk.download('stopwords') # download stopwords
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


try:
    data = load_data('pesos.csv')
except:
    data = pd.read_csv('https://word-emeddings.s3.us-west-2.amazonaws.com/20211112_JSTOR_2_full_text_embeddings_saved_COMPLETED.csv')

try:
    article_df = load_data('datajstore.csv')
except:
    article_df = pd.read_csv('https://word-emeddings.s3.us-west-2.amazonaws.com/20211024_main_article_dataframe.csv')

article_df = article_df.loc[:article_df.shape[0]-2,:]

print(data.shape)
print(article_df.shape)

def strip_punctuation_stopwords(token_list):
    return [word.lower() for word in token_list \
                if word.lower() not in nltk.corpus.stopwords.words() \
                and word not in string.punctuation \
                and len(word) > 1]

st.subheader('Documents')


# in this case weighted embedding is just "full embedding"
weighted_embedding = sentence_model.encode(text, show_progress_bar=False)
weighted_embedding2 = pd.DataFrame(weighted_embedding.T)
print(type(weighted_embedding2))
print(weighted_embedding2.shape)
weighted_embedding2.columns = data.columns
print(data.head())
print(data.tail())
print(weighted_embedding2)
print(weighted_embedding2.columns, file=sys.stderr)
data2 = pd.concat([data, weighted_embedding2]).reset_index(drop=True)
print(data2.head())
print(data2.tail())
print('DID IT MAKE SENSE?')
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
article_index = [x for x in data2.head().index if x != 1500]
dataF = article_df.loc[article_index,:].copy()
dataF['first'] = dataF.fullText.apply(lambda x: x[0:3000])
dataF['last'] = dataF.fullText.apply(lambda x: x[-3000:])
# dataF = dataF.loc[:,['abstract', 'creator', 'datePublished']]
st.dataframe(dataF)
