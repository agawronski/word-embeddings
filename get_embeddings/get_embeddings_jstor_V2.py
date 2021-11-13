#-------------------------------------------------------------------------------
# nohup python3 -u get_embeddings_wiki_V2.py > output_get_embeddings_wiki_V2.log &
# pip3 install install beautifulsoup4
# pip3 install sentence-transformers
# pip3 install pandas
from urllib import request
from io import StringIO
import json
import os
import re

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt')

sentence_model = SentenceTransformer("all-mpnet-base-v2")

data = pd.read_csv('https://word-emeddings.s3.us-west-2.amazonaws.com/20211107_main_article_dataframe_small.csv')

# test embedding the full text
# one = sentence_model.encode(data.fullText[0], show_progress_bar=False)
# >>> one.shape
# (768,)

outlist = []
for index, row in data.iterrows():
    if index % 10 == 0:
        print(index)
    if index % 200 == 0:
        print(row['fullText'])
        print('saving')
        outdf = pd.DataFrame(outlist)
        outdf.to_csv('20211112_JSTOR_2_full_text_embeddings_saved.csv', index=False)
    single_embedding = sentence_model.encode(row['fullText'], show_progress_bar=False)
    outlist.append(single_embedding)

outdf = pd.DataFrame(outlist)

outdf.to_csv('20211112_JSTOR_2_full_text_embeddings_saved_COMPLETED.csv', index=False)


#-------------------------------------------------------------------------------
