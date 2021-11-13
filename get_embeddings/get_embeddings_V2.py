#-------------------------------------------------------------------------------
# ssh -i /Users/aidangawronski/Documents/fourth_brain_capstone/fourth-brain-basic-ec2.pem ec2-user@ec2-52-13-29-251.us-west-2.compute.amazonaws.com
# nohup python3 -u get_embeddings_V2.py > output_V2.log &
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

# sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")
# https://www.sbert.net/docs/pretrained_models.html
sentence_model = SentenceTransformer("all-mpnet-base-v2")

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


def strip_punctuation_stopwords(token_list):
    return [word.lower() for word in token_list \
                if word.lower() not in nltk.corpus.stopwords.words() \
                and word not in string.punctuation \
                and len(word) > 1]


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

data.shape, len(data.identifier.unique()), len(data.identifier.unique())

data.loc[data.shape[0]] = [None]*data.shape[1]

# url = 'https://www.cbsnews.com/news/hawaiian-scientists-discover-one-of-the-youngest-planets-ever-seen/'
# output = url_2_text(url)
# output = strip_punctuation_stopwords(output)


text_from_user = """
Ancient Traces of Life Discovered Encased in a 2.5 Billion-Year-Old Ruby
While analyzing some of the world’s oldest colored gemstones, researchers from the University of Waterloo discovered carbon residue that was once ancient life, encased in a 2.5 billion-year-old ruby.
The research team, led by Chris Yakymchuk, professor of Earth and Environmental Sciences at Waterloo, set out to study the geology of rubies to better understand the conditions necessary for ruby formation. During this research in Greenland, which contains the oldest known deposits of rubies in the world, the team found a ruby sample that contained graphite, a mineral made of pure carbon. Analysis of this carbon indicates that it is a remnant of early life.
“The graphite inside this ruby is really unique. It’s the first time we’ve seen evidence of ancient life in ruby-bearing rocks,” says Yakymchuk. “The presence of graphite also gives us more clues to determine how rubies formed at this location, something that is impossible to do directly based on a ruby’s color and chemical composition.”
The presence of the graphite allowed the researchers to analyze a property called isotopic composition of the carbon atoms, which measures the relative amounts of different carbon atoms. More than 98 percent of all carbon atoms have a mass of 12 atomic mass units, but a few carbon atoms are heavier, with a mass of 13 or 14 atomic mass units.
“Living matter preferentially consists of the lighter carbon atoms because they take less energy to incorporate into cells,” said Yakymchuk. “Based on the increased amount of carbon-12 in this graphite, we concluded that the carbon atoms were once ancient life, most likely dead microorganisms such as cyanobacteria.”
The graphite is found in rocks older than 2.5 billion years ago, a time on the planet when oxygen was not abundant in the atmosphere, and life existed only in microorganisms and algae films.
During this study, Yakymchuk’s team discovered that this graphite not only links the gemstone to ancient life but was also likely necessary for this ruby to exist at all. The graphite changed the chemistry of the surrounding rocks to create favorable conditions for ruby growth. Without it, the team’s models showed that it would not have been possible to form rubies in this location.
The study, “Corundum (ruby) growth during the final assembly of the Archean North Atlantic Craton, southern West Greenland,” was recently published in Ore Geology Reviews. A companion study, “The corundum conundrum: Constraining the compositions of fluids involved in ruby formation in metamorphic melanges of ultramafic and aluminous rocks,” was published in the journal Chemical Geology in June.
References:
“Corundum (ruby) growth during the final assembly of the Archean North Atlantic Craton, southern West Greenland” by Chris Yakymchuk, Vincent van Hinsberg, Christopher L. Kirkland, Kristoffer Szilas, Carson Kinney, Jillian Kendrick and Julie A.Hollis, 20 August 2021, Ore Geology Reviews.
“The corundum conundrum: Constraining the compositions of fluids involved in ruby formation in metamorphic melanges of ultramafic and aluminous rocks” by Vincent van Hinsberg, Chris Yakymchuk, Angunguak Thomas Kleist Jepsen, Christopher L.Kirkland and Kristoffer Szilas, 20 March 2021, Chemical Geology.
"""

def get_weighted_embedding(token_list_long):
    text_token = word_tokenize(text_from_user)
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


data.loc[data.shape[0]-1,'fullText'] = ' '.join(text_from_user)

data.shape

outlist = []
for index, row in data.iterrows():
    if index % 10 == 0:
        print(index)
    if index % 200 == 0:
        print(row['fullText'])
        print('saving')
        outdf = pd.DataFrame(outlist)
        outdf.to_csv('20211020_weighted_embeddings_saved.csv')
    weighted_embedding = get_weighted_embedding(row['fullText'])
    outlist.append(weighted_embedding)


outdf = pd.DataFrame(outlist)

outdf.to_csv('20211020_weighted_embeddings_saved_FULL.csv')

# reading it in
# ting = pd.read_csv('20211017_embeddings_saved.csv', index_col=0)
# ting.head(1)



























#-------------------------------------------------------------------------------

