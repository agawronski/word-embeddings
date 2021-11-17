# Word Embeddings

1. Can we group similar client requests together? (Eg. Google News)
    - Kmeans on embeddings works reasonably well
        - embeddings with SentenceTransformer("all-mpnet-base-v2")
        - Elbow method does not provide a clear number of clusters
        - Either set the number of clusters to be:
            a) very high, for example n_clusters = (n/10), or
            b) low, for example n_clusters = 20, and get the cluster centers and find the closest ones to the cluster centers
    - Take a new request, get it's embedding and find the nearest 5 like in the app in this repo

2. Can we perform NER for unstructured data geared towards the Tech Industry or Healthcare industry with reasonable accuracy?
    - ???

3. Can we find hierarchical patterns in the topics for requests to identify temporal directions of the requests?
    - one method with Dynamic Topic Modeling

V1 JStor:
http://glg-load-balancer-380766649.us-west-2.elb.amazonaws.com:8501/

V1 Wiki:
http://glg-load-balancer-wiki-v1-1343157830.us-west-2.elb.amazonaws.com:8501/

V1 BioBert:
http://glg-load-balancer-bio-bert-v1-1743689293.us-west-2.elb.amazonaws.com:8501/



--------------------------------------------------------------------------------
get_embeddings/inital_clustering.py
- turn healthcare_industry_from_1900_2021.jsonl into a data frame
- strip punctuation and stopwords
- get the embedding for every single word of the article
- get embeddings using: SentenceTransformer("distilbert-base-nli-mean-tokens")
- get the average embedding across all words
- explored kmeans clustering

get_embeddings/get_embeddings_jstor.py
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
