import nltk.tokenize.texttiling as texttiling
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from cmath import inf
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import euclidean, cosine
import pandas as pd
from sklearn.decomposition import PCA

'''
Gets TF-IDF scores between all tiles
'''
def GetScores(tiles):
    tfidf = TfidfVectorizer()
    result = tfidf.fit_transform(tiles)
    
    # get indexing
    print('\nWord indexes:')
    print(tfidf.vocabulary_)
    
    # display tf-idf values
    print('\ntf-idf value:')
    print(result)

    return result, tfidf

def Cluster(tile_vectors):
    Sum_of_squared_distances = []
    K = range(2,tile_vectors.shape[0]//2)
    opt_clusters = None
    opt_clusters_err = float('inf')

    for k in K:
        km = KMeans(n_clusters=k, max_iter=200, n_init=10)
        km = km.fit(tile_vectors)
        Sum_of_squared_distances.append(km.inertia_)
        if km.inertia_ < opt_clusters_err:
            opt_clusters = km
            opt_clusters_err = km.inertia_

    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    return opt_clusters

def GetRelevantKeywords(n_terms, tile_vectors, cluster_labels, vectorizer):
    """This function return
def TextTiling(text):
    sentences = GetSentList(text)
    text_final = []
    for sent_in_text in sentences:
        word_lst = GetWordList(sent_in_text)
        stop_words_filtered_word_list = RemoveStopWords(word_lst)
        lemmatized_words = Lemmatize(stop_words_filtered_word_list)
        cleaned_sent = JoinWordLst(lemmatized_words)
        text_final.append(cleaned_sent)
    cleaned_text = '. '.join(text_final)
    tt = texttiling.TextTilingTokenizer()
    segments = tt.tokenize(cleaned_text)
    return segmentss the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(tile_vectors.todense()).groupby(cluster_labels).mean() # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names_out() # access tf-idf terms
    cluster_to_keywords = {}
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        top_terms = [terms[t] for t in np.argsort(r)[-n_terms:]]
        cluster_to_keywords[i] = top_terms
        print(','.join(top_terms)) # for each row of the dataframe, find the n terms that have the highest tf idf score
    return cluster_to_keywords

def GetClusterTerms(vectorizer, tile_vectors, cluster_labels, cluster_index):
  df = pd.DataFrame(tile_vectors.todense()).groupby(cluster_labels).mean() # groups the TF-IDF vector by cluster
  terms = vectorizer.get_feature_names_out() # access tf-idf terms
  for i,r in df.iterrows():
      if i == cluster_index:
        print([t for t in np.argsort(r)])
        return [terms[t] for t in r]

def GetTopKeyword(clusters, segments, tile_vectors, vectorizer):
  closest_pt_idx = []
  for iclust in range(clusters.n_clusters):
      # get all points assigned to each cluster:
      cluster_pts = [vectorizer.vocabulary_[term] for term in GetClusterTerms(vectorizer, tile_vectors, clusters.labels_, iclust)]
      # get all indices of points assigned to this cluster:
      cluster_pts_indices = np.where(clusters.labels_ == iclust)[0]

      cluster_cen = clusters.cluster_centers_[iclust]
      min_idx = np.argmin([euclidean(segments[idx], cluster_cen) for idx in cluster_pts_indices])
      print('closest point to cluster center: ', cluster_pts[min_idx])
      print('closest index of point to cluster center: ', cluster_pts_indices[min_idx])
      print('  ', segments[cluster_pts_indices[min_idx]])
      closest_pt_idx.append(cluster_pts_indices[min_idx])
  return closest_pt_idx 

# TODO: Fix this function to give the n closest vectors to each cluster center
def RankTiles(clusters: KMeans, tile_vectors, dst_algo = cosine):
    sorted_clusters = []
    for cluster_center in clusters.cluster_centers_:
        sorted_tiles_curr_cluster = sorted(tile_vectors, key=lambda tile : dst_algo(tile, cluster_center))
        sorted_clusters.append(sorted_tiles_curr_cluster)
    
    return sorted_clusters

def GetSummaryVectors(n, clusters, tile_vectors, dst_algo = cosine):
    sort_clusters = RankTiles(clusters, tile_vectors)
    summary_vectors = [cluster[:n] for cluster in sort_clusters]
    return summary_vectors