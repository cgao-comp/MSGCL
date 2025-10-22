import os
import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
import time
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

def get_degree_feature_list(edges_list_path, node_num, init='one-hot'):

    x_list = []
    max_degree = 0
    adj_list = []
    degree_list = []
    ret_degree_list = []

    edges_dir_list = sorted(os.listdir(edges_list_path))

    for edges_file in edges_dir_list:
        edges_path = os.path.join(edges_list_path, edges_file)
        adj_lilmatrix = get_adj_lilmatrix(edges_path, node_num)
        adj = sp.coo_matrix(adj_lilmatrix)
        adj_list.append(adj)

        degrees = adj.sum(axis=1).astype(int)
        max_degree = max(max_degree, degrees.max())
        degree_list.append(degrees)
        ret_degree_list.append(torch.FloatTensor(degrees).cuda() if torch.cuda.is_available() else degrees)

    for degrees in degree_list:
        if init == 'gaussian':
            fea_list = [np.random.normal(degree, 0.0001, max_degree + 1) for degree in degrees]
            fea_arr = np.array(fea_list)
            fea_tensor = torch.FloatTensor(fea_arr)
            x_list.append(fea_tensor.cuda() if torch.cuda.is_available() else fea_tensor)
        elif init == 'combine':
            fea_list = [np.random.normal(degree, 0.0001, max_degree + 1) for degree in degrees]
            fea_arr = np.array(fea_list)
            fea_arr = np.hstack((fea_arr, adj_list[i].toarray()))
            fea_tensor = torch.FloatTensor(fea_arr)
            x_list.append(fea_tensor.cuda() if torch.cuda.is_available() else fea_tensor)
        elif init == 'one-hot':
            degrees = np.asarray(degrees, dtype=int).flatten()
            one_hot_feature = np.eye(max_degree + 1)[degrees]
            x_list.append(one_hot_feature.cuda() if torch.cuda.is_available() else one_hot_feature)
        else:
            raise AttributeError('Unsupported feature initialization type!')

    return x_list, max_degree + 1, ret_degree_list

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)

    return labels_onehot

def check_and_make_path(to_make):
    if to_make == '':
        return
    if not os.path.exists(to_make):
        os.makedirs(to_make)


def get_vttdata(node_num):

    all_nodes = np.arange(node_num)
    idxes = np.random.choice(all_nodes, 20 + 44 + 147)
    idx_train, idx_val, idx_test = idxes[:20], idxes[20:-147], idxes[-147:]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return idx_train, idx_val, idx_test


def get_afldata(dice, node_num, edges_path):
    time_matrix = get_adj_lilmatrix(edges_path, node_num)
    time_matrix=time_matrix.toarray()
    similairity_matrix_structure,graph1 = getSimilariy_modified(time_matrix, node_num)
    feature_similarity_matrix = cosine_similarity(time_matrix)
    features_adj_matrix = time_matrix+0.3*similairity_matrix_structure
    features_adj = sp.coo_matrix(features_adj_matrix, dtype=float)
    adj = time_matrix+0.2*feature_similarity_matrix
    adj = sp.coo_matrix(adj)
    features_adj = normalize(features_adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj)
    return adj, features_adj, graph1

def getGraph(OneZeromatrix, node_num):
    graph = nx.Graph()
    graph.add_nodes_from(range(node_num))
    for i in range(node_num):
        for j in range(i+1,node_num):
            if OneZeromatrix[i, j] != 0:
                graph.add_edge(i, j)
    return graph


def get_adj_lilmatrix(edge_path, node_num):
    A = sp.lil_matrix((node_num, node_num), dtype=int)
    with open(edge_path, 'r') as fp:
        content_list = fp.readlines()
        for line in content_list[0:]:
            line_list = line.split(" ")
            from_id, to_id = line_list[0], line_list[1]
            if from_id == to_id:
                continue
            A[int(from_id), int(to_id)] = 1
            A[int(to_id), int(from_id)] = 1
    return A


def get_features_lilmatrix(edge_path, node_num):
    features = sp.lil_matrix((node_num, node_num), dtype=int)
    with open(edge_path, 'r') as fp:
        content_list = fp.readlines()
        for line in content_list[0:]:
            line_list = line.split(" ")
            from_id, to_id, weight = line_list[0], line_list[1], line_list[2]
            if from_id == to_id:
                continue
            features[int(to_id), int(from_id)] = weight
    return features

def getSimilariy_modified(OneZeromatrix,node_num):
    similar_matrix = sp.lil_matrix((node_num,node_num),dtype=float)
    graph = getGraph(OneZeromatrix, node_num)
    edges_list = list(graph.edges())
    node_list = list(graph.nodes())
    for i, node in enumerate(node_list):
        neibor_i_list = list(graph.neighbors(node))
        first_neighbor = neibor_i_list
        for k, second_nighbor in enumerate(first_neighbor):
            second_list = list(graph.neighbors(second_nighbor))
            neibor_i_list = list(set(neibor_i_list).union(set(second_list)))
        neibor_i_num = len(first_neighbor)
        for j, node_j in enumerate(neibor_i_list):
            neibor_j_list = list(graph.neighbors(node_j))
            neibor_j_num = len(neibor_j_list)
            commonNeighbor_list = [x for x in first_neighbor if x in neibor_j_list]
            commonNeighbor_num = len(commonNeighbor_list)
            neibor_i_num_x = neibor_i_num
            if (i,j) in edges_list:
                commonNeighbor_num = commonNeighbor_num + 2
                neibor_j_num = neibor_j_num + 1
                neibor_i_num_x = neibor_i_num + 1
            similar_matrix[node, node_j] = (2*commonNeighbor_num)/(neibor_j_num + neibor_i_num_x)
    return similar_matrix, graph


def neighbor_similarity(adjacency_matrix):

    num_nodes = adjacency_matrix.shape[0]
    similarity_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                similarity_matrix[i][j] = 1
                continue
            neighbor_i = np.where(adjacency_matrix[i] == 1)[0]
            neighbor_j = np.where(adjacency_matrix[j] == 1)[0]
            common_neighbors = len(set(neighbor_i).intersection(neighbor_j))
            similarity = common_neighbors / np.sqrt((len(neighbor_i)+1) * (len(neighbor_j))+1)
            similarity_matrix[i][j] = similarity

    return similarity_matrix


def pearson_similarity(adj_matrix):
    degrees = np.sum(adj_matrix, axis=1)
    mean_neighbor_degrees = np.dot(adj_matrix, degrees) / np.sum(adj_matrix, axis=1)
    corrected_adj_matrix = adj_matrix - np.outer(degrees, mean_neighbor_degrees) / (
                np.sum(adj_matrix) - np.sum(degrees))
    n = corrected_adj_matrix.shape[0]
    pearson_sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if np.all(np.logical_and(corrected_adj_matrix[i] != 0, corrected_adj_matrix[j] != 0)):
                pearson_sim_matrix[i, j] = np.sum(corrected_adj_matrix[i] * corrected_adj_matrix[j]) \
                                           / (np.linalg.norm(corrected_adj_matrix[i]) * np.linalg.norm(
                    corrected_adj_matrix[j]))
                pearson_sim_matrix[j, i] = pearson_sim_matrix[i, j]  
    np.fill_diagonal(pearson_sim_matrix, 1) 
    return pearson_sim_matrix



def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(outputs, labels):
    preds = outputs.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def binary_accuracy(outputs, labels):
    preds = outputs.gt(0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def check_and_creat_dir(file_url):
    file_gang_list = file_url.split('/')
    if len(file_gang_list) > 1:
        [fname, fename] = os.path.split(file_url)
        if not os.path.exists(fname):
            os.makedirs(fname)
        else:
            return None

    else:
        return None


class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def preprocess_graph(adj, layer, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    reg = [1] * (layer)

    adjs = []
    for i in range(len(reg)):
        adjs.append(ident - (reg[i] * laplacian))

    return adjs


def laplacian(adj):
    rowsum = np.array(adj.sum(1))
    degree_mat = sp.diags(rowsum.flatten())
    lap = degree_mat - adj
    return torch.FloatTensor(lap.toarray())
