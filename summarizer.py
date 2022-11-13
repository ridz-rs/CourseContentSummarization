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

        


def GetSummarizedTiles(tiles, clusters, cluster_to_keywords):
    summarized_tiles = ''
    for itile in range(len(tiles)):
        target_cluster_index = clusters.labels_[itile]
        # print(target_cluster_index)
        top_keywords = cluster_to_keywords[target_cluster_index]
        summarized_tiles += FilterSents(tiles[itile], top_keywords, n_sents=1)
        summarized_tiles += '. '
    
    return summarized_tiles