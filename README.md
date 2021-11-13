# Word Embeddings 

get_embeddings/inital_clustering.py
- turn healthcare_industry_from_1900_2021.jsonl into a data frame
- strip punctuation and stopwords
- get the embedding for every single word of the article
- get embeddings using: SentenceTransformer("distilbert-base-nli-mean-tokens")
- get the average embedding across all words


get_embeddings/get_embeddings_V3.py
- uses function "get_weighted_embedding"
- get word tokens using word_tokenize from nltk.tokenize
- strip punctuation (string.punctuation) and remove stopwords (nltk.corpus.stopwords)
- get token frequencies (nltk.probability.FreqDist)
- keep the top 30 most used tokens
- get the weight of each token (token frequency/total count top 30)
- get the embedding using: SentenceTransformer("all-mpnet-base-v2")
- get the average embedding, averaged using the weighting


get_embeddings/get_embeddings_wiki.py
- same as get_embeddings_V3.py but gets the embeddings for people_wiki.csv