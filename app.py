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

if data is None:
    data = pd.read_csv('https://word-emeddings.s3.us-west-2.amazonaws.com/20211020_weighted_embeddings_saved_FULL.csv')

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
    final = weighted_embed.sum(axis=0)
    final2 =  pd.DataFrame(final)
    return final2.to_html()


@app.route('/')
def my_form():
    df_html = data.describe().to_html()
    return render_template('my-form.html', df_html=df_html)


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    weighted_embedding = get_weighted_embedding(text)
    print(data.shape)
    print(weighted_embedding)
    data2 = pd.concat(data, weighted_embedding)
    print(data2.shape)
    return render_template('my-form.html', df_html=weighted_embedding)


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