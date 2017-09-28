import numpy as np

def _davies_boudin_score(doc_vecs, swarm_config, cluster_centers):
    """ Returns Davies-Boudin score for each configuration
    :param swarm_config: Map of cluster ids to their configuration
    :param cluster_centers: Map of cluster ids to their geometrical centers
    :return: Score for the current swarm configuration
    """
    cluster_config_score = []
    for cluster_id, docs in swarm_config.iteritems():
        if docs:
            _doc_vecs = [doc_vecs[doc_id] for doc_id in docs]
            cluster_config_score.append(
                sum(map(lambda a: np.linalg.norm(a - cluster_centers[cluster_id]), _doc_vecs)))
        else:
            cluster_config_score.append(0)
    _score = 0
    for id_base, base_cluster_config_score in enumerate(cluster_config_score):
        _score += max([((base_cluster_config_score + iterative_cluster_config_score) / float(
            np.linalg.norm(cluster_centers[id_base + 1] - cluster_centers[id_iter + 1]))) for
                       id_iter, iterative_cluster_config_score in enumerate(cluster_config_score) if
                       id_base != id_iter])
    return _score / np.shape(doc_vecs)[0]
