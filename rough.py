from text_tiling import TextTiling
from clustering import GetScores, Cluster, GetSummaryVectors

with open('C:/Users/Riddhesh/Desktop/remove_small_sents.txt', 'r') as f:
    text = f.read() 
    segments = TextTiling(text)

with open('C:/Users/Riddhesh/Desktop/TT.txt', 'w') as f: 
    # for segment in segments:
    f.writelines(segments)

tile_vectors = GetScores(segments)
clusters = Cluster(tile_vectors)
summary_vectors = GetSummaryVectors(3, clusters, tile_vectors)
print(summary_vectors)
