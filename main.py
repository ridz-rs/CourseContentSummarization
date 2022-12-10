from text_tiling import TextTiling
from clustering import *
from summarizer import GetSummarizedTiles, TileSelectSummary

TEXT_FILE_PATH = 'PS3CSC373.txt'
OUT_PATH = 'TT.txt'
BLOCK_COMPARISON, VOCABULARY_INTRODUCTION = 0, 1 
BERT_ENCODING_SPACE_COMPARISON, SBERT_ENCODING_SPACE_COMPARISON, PHRASE_BERT_ENCODING_SPACE_COMPARISON= 2, 3, 4
method = BERT_ENCODING_SPACE_COMPARISON

is_bert = True

with open(TEXT_FILE_PATH, 'r') as f:
    text = f.read() 
    segments, tt = TextTiling(text, method)

with open(OUT_PATH, 'w') as f: 
    # for segment in segments:
    f.writelines(segments)

if not is_bert:
  # tfidf
  tile_vectors, vectorizer = GetScores(segments)
  clusters = Cluster(tile_vectors)
  cluster_to_keywords = GetRelevantKeywords(3, tile_vectors, clusters.labels_, vectorizer)
  summarized_tiles = GetSummarizedTiles(segments, clusters, cluster_to_keywords)
  print(summarized_tiles)
else:
  # Add bert summarization pipeline
  tile_vectors = GetBERTTileScores(tt, segments)
  clusters = Cluster(tile_vectors)
  tile_to_vec = {}
  for i, seg in enumerate(segments):
    tile_to_vec[seg] = tile_vectors[i]
  summarized_tiles = TileSelectSummary(clusters, tile_vectors, tile_to_vec)
  print(summarized_tiles)