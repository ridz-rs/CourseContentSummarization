from cmath import inf
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import euclidean, cosine
import pandas as pd

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

    return result

def Cluster(tile_vectors):
    Sum_of_squared_distances = []
    K = range(2,tile_vectors.shape[0])
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
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(tile_vectors.todense()).groupby(cluster_labels).mean() # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names_out() # access tf-idf terms
    cluster_to_keywords = {}
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        top_terms = [terms[t] for t in np.argsort(r)[-n_terms:]]
        cluster_to_keywords[i] = top_terms
        print(','.join(top_terms)) # for each row of the dataframe, find the n terms that have the highest tf idf score
    return cluster_to_keywords

def FilterSents(text, top_keywords, n_sents=2):
    text = text.lower()
    text = text.replace('.', '<SEP>')
    text = text.replace('!', '<SEP>')
    text = text.replace('?', '<SEP>')
    sent_list = text.split('<SEP>')
    avg_scores = []
    for isent in range(len(sent_list)):
        score = 0
        for word in sent_list[isent].split(' '):
            if word in top_keywords:
                score += 1
        avg_scores.append(score/len(sent_list[isent]))
    return '. '.join(list(np.array(sent_list)[np.argsort(avg_scores)])[:n_sents])

        


def GetSummarizedTiles(tiles, clusters, cluster_to_keywords):
    summarized_tiles = ''
    for itile in range(len(tiles)):
        target_cluster_index = clusters.labels_[itile]
        print(target_cluster_index)
        top_keywords = cluster_to_keywords[target_cluster_index]
        summarized_tiles += FilterSents(tiles[itile], top_keywords)
    
    return summarized_tiles