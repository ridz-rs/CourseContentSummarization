import numpy as np

def FilterSents(text, top_keywords, n_sents=2):
    text = text.lower()
    text = text.replace('.', '<SEP>')
    text = text.replace('!', '<SEP>')
    text = text.replace('?', '<SEP>')
    sent_list = text.split('<SEP>')
    avg_scores = []
    for isent in range(len(sent_list)):
        if not sent_list[isent]:
          continue
        score = 0
        for word in sent_list[isent].split(' '):
            if word in top_keywords:
                score += 1
        avg_scores.append(score/len(sent_list[isent]))
    return '. '.join(list(np.array(sent_list)[np.argsort(avg_scores)])[:n_sents])

'''
Generate summary text by filtering sentences 
according to which sentences have highest word scores
'''
def GetSummarizedTiles(tiles, clusters, cluster_to_keywords):
    summarized_tiles = ''
    for itile in range(len(tiles)):
        target_cluster_index = clusters.labels_[itile]
        # print(target_cluster_index)
        top_keywords = cluster_to_keywords[target_cluster_index]
        summarized_tiles += FilterSents(tiles[itile], top_keywords, n_sents=1)
        summarized_tiles += '. '
    
    return summarized_tiles

'''
Creating summary by finding closest tiles from cluster centroids
'''
def TileSelectSummary(clusters, tile_vectors):
    summary_tiles = []
    for iclust in range(clusters.n_clusters):
    cluster_pts_indices = np.where(clusters.labels_ == iclust)[0]
    cluster_tiles = tile_vectors[cluster_pts_indices]
    closest_tile_vec = get_closest_tile_vec(cluster_pts_indices, tile_vectors, clusters.cluster_centers_[iclust])
    for key, value in tile_to_vec.items():
        if (value==closest_tile_vec).all():
        summary_tiles.append(key)
        break

    return ' '.join(summary_tiles)
