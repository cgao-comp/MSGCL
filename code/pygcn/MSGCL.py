from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import assment_result
import os
from collections import Counter
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils_MSGCL import *
from models import MSGCL

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='none')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--origin', action='store_true', default=True,
                    help='Keep the original implementation as the paper.')
parser.add_argument('--test_only', action="store_true", default=False,
                    help='Test on existing model')
parser.add_argument('--repeat', type=int, default=3,
                    help='number of experiments')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Show training process')
parser.add_argument('--split', type=str, default='random',
                    help='Data split method')
parser.add_argument('--rho', type=float, default=0.1,
                    help='Adj matrix corruption rate')
parser.add_argument('--corruption', type=str, default='node_shuffle',
                    help='Corruption method')
parser.add_argument('--gnnlayers', type=int, default=1, help="Number of gnn layers")


def get_data(dice, node_num, time_edges_path):
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("prepared for loading data!")
    adj, features, graph = get_afldata(dice, node_num, time_edges_path)
    print("Load have done!")
    idx_train, idx_val, idx_test = get_vttdata(node_num)

    if args.cuda:
        features = features.cuda()
        adj = adj.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    return adj, features, idx_train, idx_val, idx_test, args, graph


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def train(num_epoch, time_step, last_embedding, patience=30, verbose=False):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")
    best_epoch = -1
    #负对的候选集
    # neg_matrix = torch.ones(features.size()[0], features.x.size()[0]).to(device)
    for epoch in range(num_epoch):
        t = time.time()
        optimizer.zero_grad()
        if time_step >= 1:
            H, outputs, labels = model(features, adj, last_embedding)
        else:
            H,outputs, labels = model(features, adj)
        if args.cuda:
            labels = labels.cuda()

        time_matrix_array = time_matrix+np.eye(len(time_matrix))
        time_matrix_array = torch.FloatTensor(time_matrix_array)
        loss_train_multi = F.mse_loss(H, time_matrix_array)

        loss_train = F.binary_cross_entropy_with_logits(outputs, labels)

        loss = 0.2*loss_train + 0.4*loss_train_multi
        acc_train = binary_accuracy(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_final = loss.item()
        accuracy = acc_train.item()
        if verbose:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss),
                  'acc_train: {:.4f}'.format(accuracy),
                  'time: {:.4f}s'.format(time.time() - t))
        if loss_final < best_loss:
            best_loss = loss_final
            best_epoch = epoch
        if epoch == best_epoch + patience:
            break


def test(verbose=False):
    with torch.set_grad_enabled(False):
        model.eval()
        last_embedding = 0
        outputs1, outputs2, weight1, weight2 = model(features, adj, last_embedding)
        outputs = (outputs1 + outputs2) / 2
        weight = (weight1 + weight2) / 2
        outputs_numpy = outputs.data.cpu().numpy()


    return outputs, weight, outputs_numpy


def get_matrix(edge_path, node_num):
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

if __name__ == "__main__":

    dataset = "java"
    node_num = 376
    dice = 0.0
    time_weight_list = [0]
    embedding_list = [0]
    NMI_list = []
    Q_list = []
    ARI_list = []
    NMI_t = []
    ARI_t = []
    Q_t = []

    hidden = node_num

    islabel = True
    method = "MSGCL"
    base_data_path = "/sda/home/tengmin/code/MSGCL/dataset/"
    edges_base_path = "/edges"
    label_base_path = "/labels"
    edges_data_path = base_data_path + dataset + edges_base_path
    file_num = int(len(os.listdir(edges_data_path))/2)
    print("file_num:{}".format(file_num))

    for t in range(file_num):
        print("The {}th snapshot".format(t + 1))
        time_edges_path = edges_data_path + "/edges_" + "t" + str(t + 1) + "_1" + ".txt"
        time_matrix = get_matrix(time_edges_path, node_num)
        time_matrix = time_matrix.toarray()
        adj, features, idx_train, idx_val, idx_test, args, graph = get_data(dice, node_num, time_edges_path)
        print("Get data done!")
        features = (features + sp.eye(features.shape[0])).toarray()
        features = torch.FloatTensor(features)
        adj_1st = (adj + sp.eye(adj.shape[0])).toarray()
        adj = torch.FloatTensor(adj_1st)
        if islabel:
            label_data_path = base_data_path + dataset + label_base_path + "/label_" + str(t + 1) + "_1" + ".txt"
            original_cluster = np.loadtxt(label_data_path, dtype=int)
        sumARI = 0
        sumNMI = 0
        sumQ = 0
        for i in range(args.repeat):
            model = MSGCL(num_feat=features.shape[1],
                         num_hid=node_num,
                         time_step=t,
                         graph=graph,
                         time_weight=time_weight_list[-1],
                         dropout=args.dropout,
                         rho=args.rho,
                         corruption=args.corruption)
            print("----- %d / %d runs -----" % (i, args.repeat))
            t_total = time.time()
            if args.test_only:
                model = torch.load("model")
            else:
                train(args.epochs, t, embedding_list[-1], verbose=args.verbose)
                print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
            outputs, weight, outputs_numpy = test(verbose=args.verbose)
            save_emb_path = "./code/embeddings/" + dataset + "/" + method + "/embeddingF_" + str(
                t) + ".txt"
            check_and_creat_dir(save_emb_path)
            np.savetxt(save_emb_path, outputs_numpy, fmt="%f")
            time_weight_list.append(weight)
            embedding_list.append(outputs)
            if islabel:
                k = len(Counter(original_cluster))
                print(k)
                NMI, ARI, Q = assment_result.eva(original_cluster, outputs_numpy, k, time_matrix)
                print("NMI value is：{}".format(NMI))
                NMI_list.append(NMI)
                print("ARI value is：{}".format(ARI))
                ARI_list.append(ARI)
                print("Q value is：{}".format(Q))
                Q_list.append(Q)
                sumNMI += NMI
                sumARI += ARI
                sumQ += Q
        t_ARI = sumARI / args.repeat
        t_NMI = sumNMI / args.repeat
        t_Q = sumQ / args.repeat
        ARI_t.append(t_ARI)
        NMI_t.append(t_NMI)
        Q_t.append(t_Q)
    if islabel:
        ave_NMI = np.mean(NMI_list)
        print('--------------------------------------------')
        print("The average NMI is:{}".format(ave_NMI))
        ave_ARI = np.mean(ARI_list)
        print('--------------------------------------------')
        print("The average ARI is:{}".format(ave_ARI))
        ave_Q = np.mean(Q_list)
        print('--------------------------------------------')
        print("The average Q is:{}".format(ave_Q))


