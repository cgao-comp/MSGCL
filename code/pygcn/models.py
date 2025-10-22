import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.cluster import KMeans
import random
import scipy.sparse as sp


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, origin=False):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

        ###
        self.origin = origin

    def forward(self, x, adj):
        if self.origin:
            x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class MSGCL(nn.Module):
    def __init__(self, num_feat, num_hid, time_step, graph, time_weight, dropout, rho=0.1, readout="average", corruption="node_shuffle"):
        super(MSGCL, self).__init__()
        self.time_time =time_step
        self.time_weight_weight = time_weight
        self.graph = graph
        self.gc1 = GraphConvolution(num_feat, num_hid, time_step, time_weight)
        self.fc1 = nn.Linear(num_hid, num_hid, bias=False)
        self.gc2 = GraphConvolution(num_feat, num_hid, time_step, time_weight)
        self.fc2 = nn.Linear(num_hid, num_hid, bias=False)
        self.dropout = dropout
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

        self.rho = rho

        self.readout = getattr(self, "_%s" % readout)
        self.corruption1 = getattr(self, "_%s" % corruption)
        self.corruption2 = getattr(self, "_%s" % corruption)


    def forward(self, X, A, last_embedding=0):
        A = torch.FloatTensor(A)
        X=torch.FloatTensor(X)
        x1 = F.dropout(X, self.dropout, training=self.training)
        HHH1, weight1 = self.gc1(x1, A)
        H1 = self.prelu1(HHH1)
        # print(H1.shape)
        x2 = F.dropout(X, self.dropout, training=self.training)
        HHH2, weight2 = self.gc2(x2, A)
        H2 = self.prelu2(HHH2)
        if not self.training:
            return H1,H2,weight1,weight2
        neg_X1, neg_A1 = self.corruption1(X, A)
        x1 = F.dropout(neg_X1, self.dropout, training=self.training)
        neg_HHH1, neg_weight1 = self.gc1(x1, neg_A1)
        neg_H1 = self.prelu1(neg_HHH1)

        neg_X2, neg_A2 = self.corruption2(X, A)
        x2 = F.dropout(neg_X2, self.dropout, training=self.training)
        neg_HHH2, neg_weight2 = self.gc2(x2, neg_A2)
        neg_H2 = self.prelu2(neg_HHH2)
        s1 = self.readout(H1)
        x1 = self.fc1(s1)
        x1 = torch.mv(torch.cat((H1, neg_H1)), x1)
        labels1 = torch.cat((torch.ones(X.size(0)), torch.zeros(neg_X1.size(0))))
        s2 = self.readout(H2)
        x2 = self.fc2(s2)
        x2 = torch.mv(torch.cat((H2, neg_H2)), x2)
        labels2 = torch.cat((torch.ones(X.size(0)), torch.zeros(neg_X2.size(0))))
        x = (x1 + x2) / 2
        labels = (labels1 + labels2) / 2
        H = (H1 + H2) / 2
        H_=H @ H.T

        return H_,x,labels

    def _average(self, features):
        x = features.mean(0)
        return F.sigmoid(x)

    def _node_shuffle(self, X, A):
        perm = torch.randperm(X.size(0))
        neg_X = X[perm]
        return neg_X, A

    def euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist.sum()

    def _adj_corrupt(self, X, A):
     
        num_nodes = A.size(0)
        neg_A = A.clone()
        num_edges_to_add = min(num_nodes * 2, num_nodes * num_nodes - num_nodes - int(A.sum().item()))
        all_node_pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        random.shuffle(all_node_pairs)
        added_edges = 0
        for node1, node2 in all_node_pairs:
            if added_edges >= num_edges_to_add:
                break
            if neg_A[node1, node2] == 0:
                neg_A[node1, node2] = 1
                neg_A[node2, node1] = 1
                added_edges += 1

        return X, neg_A

    def modularity_generator(self,G):

        degrees = nx.degree(G)
        e_count = len(nx.edges(G))
        modu = np.array(
            [[float(degrees[node_1] * degrees[node_2]) / (2 * e_count) for node_1 in nx.nodes(G)] for node_2 in
             tqdm(nx.nodes(G))], dtype=np.float64)
        return modu


    def get_idcatematrix(self,node_num,k,embeddings):
        clf = KMeans(k)
        y_pred = clf.fit_predict(embeddings.detach().numpy())
        print("y_pred type is:{0}, shape is:{1}".format(type(y_pred), y_pred.shape))
        y_pred = torch.LongTensor(y_pred)
        ones = torch.sparse.torch.eye(k)
        y_one_hot = ones.index_select(0,y_pred)
        print("y_one_hot:{}".format(y_one_hot.size))
        return y_one_hot

    def get_exitembeddings(self, graph, embedding):
        exit_embeddings = []
        exitNode_list = sorted(list(graph.nodes()))
        for j, en in enumerate(embedding.detach().numpy()):
            if (j in exitNode_list):
                exit_embeddings.append(en)
        exit_embeddings = np.mat(exit_embeddings)
        return exit_embeddings
