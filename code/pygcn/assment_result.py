from sklearn.cluster import KMeans
from sklearn import metrics
import warnings
import numpy as np
from sklearn.metrics import adjusted_rand_score
warnings.filterwarnings("ignore")
from sklearn.cluster import SpectralClustering
import networkx as nx
def getPred(embeddings,k):
    a = 0
    sum = 0
    while a < 10:
        clf = KMeans(k)
        y_pred = clf.fit_predict(embeddings)

        c = y_pred.T
        epriment_cluster = c
        return epriment_cluster

def eva(labels,embeddings,k,adj):
    origin_cluster = labels
    sum_nmi = 0
    sum_ari=0
    sum_Q = 0
    graph=nx.from_numpy_array(adj)
    for _ in range(10):
        clf = KMeans(n_clusters=k, random_state=42)
        y_pred = clf.fit_predict(embeddings)
        nmi = metrics.normalized_mutual_info_score(origin_cluster, y_pred)
        sum_nmi += nmi
        ari = metrics.adjusted_rand_score(origin_cluster, y_pred)
        sum_ari += ari
        Q = getQ(graph, y_pred, adj)
        sum_Q=sum_Q+Q
        
        
    average_nmi = sum_nmi / 10
    average_ari = sum_ari / 10
    average_q = sum_Q / 10
    return average_nmi, average_ari, average_q


def getQ(G, labels, matirx_path):
    A = matirx_path
    node_num = len(G.nodes)
    edges_num = len(G.edges)
    sum = 0
    for i, d_i in enumerate(G.degree()):
        for j, d_j in enumerate(G.degree()):
            if labels[i] == labels[j]:
                sum += A[i][j] - d_i[1] * d_j[1] / (2 * edges_num)
    Q = sum / (2 * edges_num)
    return Q

