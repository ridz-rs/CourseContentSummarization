from cmath import inf
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import euclidean, cosine

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