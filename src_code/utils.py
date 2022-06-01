import os
import json
import glob
import copy
import torch
import pickle
import time

def convert_seq_to_mat(sequence):
    seq_len = len(sequence['words'])

    # initializing a matrix of NxN
    adj_mat = np.zeros((seq_len, seq_len))

    for w in sequence['words']:
        w_id = int(w['id'])
        head = int(w['head'])

        # ignore the root(0) - (-1) connection
        if head == -1:
            continue
        adj_mat[head][w_id] = 1

    return adj_mat


def adj_mat_to_tensor(mat):
    out = [0] * mat.shape[0]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            if (mat[i][j] == 1):
                out[j] = i

    return (torch.LongTensor(out))