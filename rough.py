from text_tiling import TextTiling
from clustering import *
from summarizer import GetSummarizedTiles

BLOCK_COMPARISON, VOCABULARY_INTRODUCTION = 0, 1 
BERT_ENCODING_SPACE_COMPARISON, SBERT_ENCODING_SPACE_COMPARISON, PHRASE_BERT_ENCODING_SPACE_COMPARISON= 2, 3, 4
method = 2
with open('PS3CSC373.txt', 'r') as f:
    text = f.read() 
    segments = TextTiling(text, method)

with open('TT.txt', 'w') as f: 
    # for segment in segments:
    f.writelines(segments)

# tfidf
tile_vectors, vectorizer = GetScores(segments)
clusters = Cluster(tile_vectors)
cluster_to_keywords = GetRelevantKeywords(3, tile_vectors, clusters.labels_, vectorizer)
summarized_tiles = GetSummarizedTiles(segments, clusters, cluster_to_keywords)
print(summarized_tiles)

# Add bert summarization pipeline