import json
import os
import copy
import torch
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

from time import gmtime, strftime

from torch.autograd import Variable

from gensim.models import Word2Vec


def pretrain_word_embedding(data, word_embed_len=100, pos_embed_len=20):
    corpus_words = []
    corpus_pos = []
    all_words = []
    all_pos = []
    temp = {}
    labels_file = "/content/drive/MyDrive/Dependency_Parsing/UD_English-Atis/en_atis-ud-labels.json"

    # Collect all the words in the corpus and their position
    for d in data:
        ws = []
        ps = []

        for w in d['words']:
            ws.append(w['form'])
            ps.append(w['xpostag'])
            l = w['deprel']

            if l not in temp:
                temp[l] = len(l)

        corpus_words.append(ws)
        corpus_pos.append(ps)
        all_words.extend(ws)
        all_pos.extend(ps)

    # dumping the labels in labels file
    with open(labels_file, 'w+') as f_handle:
        f_handle.write(json.dumps(temp, indent=4))

    word_to_idx = {w: idx for idx, w in enumerate(all_words)}
    pos_to_idx = {p: idx for idx, p in enumerate(all_pos)}

    # Pretrain word embeddings and positional embeddings using Word2Vec
    word_embedding = Word2Vec(corpus_words, size=word_embed_len, window=5, min_count=1, workers=8)
    pos_embedding = Word2Vec(corpus_pos, size=pos_embed_len, window=5, min_count=1, workers=8)

    # fetch the embedding and map it with each word
    final_word_embedding = torch.FloatTensor(max(word_to_idx.values()) + 1, word_embed_len)
    final_pos_embedding = torch.FloatTensor(max(pos_to_idx.values()) + 1, pos_embed_len)

    for w in word_to_idx.keys():
        idx = word_to_idx[w]
        final_word_embedding[idx, :] = torch.from_numpy(word_embedding[w])

    for p in pos_to_idx.keys():
        idx = pos_to_idx[p]
        final_pos_embedding[idx, :] = torch.from_numpy(pos_embedding[p])

    return word_to_idx, pos_to_idx, temp, final_word_embedding, final_pos_embedding


def train():
    data = json.load(open(train_data_processed_file_path, 'r'))

    # initialize word embeds
    word_embed_len = 100
    pos_embed_len = 20

    # fetch word and pos embeddings
    word_to_idx, pos_to_idx, lab_to_idx, pretrained_word_embedding, pretrained_pos_embedding = pretrain_word_embedding(
        data, word_embed_len, pos_embed_len)

    lab_len = len(lab_to_idx)
    # initialize the model
    # model = DP(word_to_idx=word_to_idx, pos_to_idx=pos_to_idx, pos_embed=pretrained_word_embedding, pos_embed=pretrained_word_embedding,word_embed_len=word_embed_len, pos_embed_len=pos_embed_len, n_labels=lab_len)
    model = DP(word_to_idx, pos_to_idx, pretrained_word_embedding, pretrained_pos_embedding, word_embed_len,
               pos_embed_len, n_labels=lab_len)

    criterion = nn.CrossEntropyLoss()
    lr = LEARNING_RATE
    weight_decay = WEIGHT_DECAY
    betas = (BETA_VAL, BETA_VAL)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    # Start Training
    for ep in range(N_EPOCHS):
        arc_losses = []
        label_losses = []
        total_losses = []

        for i in range(len(data)):
            model.zero_grad()

            # fetch the length of each sequence
            seq_len = len(data[i]['words'])

            # create the adj matrix and tree
            gold_mat = convert_seq_to_mat(data[i])
            gold_tree = adj_mat_to_tensor(gold_mat)

            arc_target = Variable(gold_tree, requires_grad=False)
            labels_target = torch.LongTensor(seq_len)

            for j, w in enumerate(data[i]['words']):
                labels_target[j] = lab_to_idx[w['deprel']]
            labels_target = Variable(labels_target, requires_grad=False)

            # prepare input
            seq = torch.LongTensor(seq_len, 3)
            for j, w in enumerate(data[i]['words']):
                seq[j, 0] = word_to_idx[w['form']]
                seq[j, 1] = pos_to_idx[w['xpostag']]
                seq[j, 2] = gold_tree[j]
            seq_var = Variable(seq)

            # run the model
            adj_mat, label_pred = model(seq_var)

            # determine losses
            arc_pred = torch.t(adj_mat)  # cross entropy loss wants the classes in the second dimension
            arc_loss = criterion(arc_pred, arc_target)
            label_loss = criterion(label_pred, labels_target)
            total_loss = arc_loss + label_loss

            # print(arc_loss)
            # print(arc_loss.item())
            arc_losses.append(arc_loss.item())
            label_losses.append(label_loss.item())
            total_losses.append(total_loss.item())

            # backprop
            total_loss.backward()
            optimizer.step()

        model.arc_loss.append(np.mean(arc_losses))
        model.label_loss.append(np.mean(label_losses))
        model.total_loss.append(np.mean(label_losses))

        if (ep % EVERY == 0):
            print("Epoch: " + str(ep))
            print("ARC loss is : " + str(np.mean(np.array(arc_losses))))
            print("Label loss is : " + str(np.mean(np.array(label_losses))))
            print("Total loss is : " + str(np.mean(np.array(label_losses))))

        if (ep % PLOT_WEIGHTS_EVERY == 0 or ep == 0):
            print("Epoch: " + str(ep))
            plt.clf()
            if ep == 0:
                plt.imshow(gold_mat)
            else:
                temp = torch.t(F.softmax(torch.t(adj_mat))).data
                plt.imshow(temp.numpy())
            plt.colorbar()
            plt.show()

    torch.save(model.state_dict(),
               "./content/drive/MyDrive/Dependency_Parsing/UD_English-Atis/dependency_parser_model.pt")
    return model


