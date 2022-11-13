from text_tiling import TextTiling
from clustering import *
from summarizer import GetSummarizedTiles

with open('C:/Users/Riddhesh/Desktop/remove_small_sents.txt', 'r') as f:
    text = f.read() 
    segments = TextTiling(text)

with open('C:/Users/Riddhesh/Desktop/TT.txt', 'w') as f: 
    # for segment in segments:
    f.writelines(segments)

tile_vectors, vectorizer = GetScores(segments)
clusters = Cluster(tile_vectors)
cluster_to_keywords = GetRelevantKeywords(3, tile_vectors, clusters.labels_, vectorizer)
summarized_tiles = GetSummarizedTiles(segments, clusters, cluster_to_keywords)
print(summarized_tiles)
