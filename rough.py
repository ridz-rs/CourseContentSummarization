from text_tiling import TextTiling
from clustering import *
from summarizer import GetSummarizedTiles

with open('PS3CSC373.txt', 'r') as f:
    text = f.read() 
    segments = TextTiling(text, 2)

with open('TT.txt', 'w') as f: 
    # for segment in segments:
    f.writelines(segments)

tile_vectors, vectorizer = GetScores(segments)
clusters = Cluster(tile_vectors)
cluster_to_keywords = GetRelevantKeywords(3, tile_vectors, clusters.labels_, vectorizer)
summarized_tiles = GetSummarizedTiles(segments, clusters, cluster_to_keywords)
print(summarized_tiles)
