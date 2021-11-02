# http://ec2-52-13-29-251.us-west-2.compute.amazonaws.com:5000/
from urllib import request
from io import StringIO
import json
import sys
import os
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from flask import Flask, request, render_template
from nltk.probability import FreqDist
from nltk.stem import LancasterStemmer
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
# from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")
# https://www.sbert.net/docs/pretrained_models.html
sentence_model = SentenceTransformer("all-mpnet-base-v2")


app = Flask(__name__)

data = None
article_df = None

if data is None:
    #data = pd.read_csv('https://word-emeddings.s3.us-west-2.amazonaws.com/20211017_embeddings_saved_FULL.csv', index_col=0)
    data = pd.read_csv('https://word-emeddings.s3.us-west-2.amazonaws.com/20211031_weighted_embeddings_saved_PARTIAL.csv')

if article_df is None:
    article_df = pd.read_csv('https://word-emeddings.s3.us-west-2.amazonaws.com/20211024_main_article_dataframe.csv')

# The last embedding was not part of the dataset originally, getting rid of it
article_df = article_df.loc[:article_df.shape[0]-2,:]


print(data.shape)
print(article_df.shape)

def strip_punctuation_stopwords(token_list):
    return [word.lower() for word in token_list \
                if word.lower() not in nltk.corpus.stopwords.words() \
                and word not in string.punctuation \
                and len(word) > 1]


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


@app.route('/')
def my_form():
    df_html = data.describe().to_html()
    return render_template('my-form.html',
                            df_html=df_html,
                            df_shape='1')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    print('HERE IS THE POST')
    print(text)
    weighted_embedding = get_weighted_embedding(text)
    weighted_embedding2 = weighted_embedding.T
    weighted_embedding2.columns = data.columns
    print(data.head())
    print(data.tail())
    print(weighted_embedding2)
    print(weighted_embedding2.columns, file=sys.stderr)
    data2 = pd.concat([data, weighted_embedding2]).reset_index(drop=True)
    print(data2.head())
    print(data2.tail())
    print('DID IT MAKE SENSE?')
    # CLUSTERING
    #kmeans = KMeans(n_clusters = 20, random_state = 1111)
    #clusters = kmeans.fit_predict(data2)
    #data2['clusters'] = clusters
    #print(data2.clusters.value_counts(), file=sys.stderr)
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
    article_index = [x for x in data2.head().index if x != 1200]
    dataF = article_df.loc[article_index,:].copy()
    dataF['first'] = dataF.fullText.apply(lambda x: x[0:3000])
    dataF['last'] = dataF.fullText.apply(lambda x: x[-3000:])
    dataF = dataF.loc[:,['abstract', 'creator', 'datePublished', 'docType', 'first', 'last', 'id',
       'identifier', 'isPartOf', 'outputFormat', 'provider', 'publicationYear',
       'title', 'url', 'wordCount', 'language', 'pageCount', 'pageEnd', 'pageStart',
       'pagination', 'sourceCategory', 'subTitle', 'tdmCategory', 'collection',
       'hasPartTitle', 'publisher']]
    #for index, row in data2.head(10).iterrows():
    #    print('----------------')
    #    print(index)
    #    print(f'Article')    
    #    ting = article_df.iloc[index,:]
    #    print(ting['fullText'][0:1000])
    #print(data2.head().index)
    return render_template('my-form.html',
                            df_html=dataF.to_html())


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')


# from flask import Flask
#
# app = Flask(__name__)
#
# @app.route('/')
# def index():
#     return 'Web App with Python Flask!'
#
# app.run(host='0.0.0.0')
