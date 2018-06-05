# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import random

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from torch.autograd import Variable

plt.switch_backend('agg')

from ttw.data_loader import Map, create_obs_dict, load_features, ResnetFeatures, FasttextFeatures, TextrecogFeatures, to_variable
from ttw.logger import create_logger

neighborhoods = ['fidi', 'uppereast', 'eastvillage', 'williamsburg', 'hellskitchen']
landmarks = Map(neighborhoods)

def create_split(Xs, ys):
    random.seed(1)
    train_Xs = {k: list() for k in Xs.keys()}
    train_ys = list()
    valid_Xs = {k: list() for k in Xs.keys()}
    valid_ys = list()

    for i in range(len(ys)):
        if random.random() > 0.7:
            for k in Xs.keys():
                valid_Xs[k].append(Xs[k][i])
            valid_ys.append(ys[i])
        else:
            for k in Xs.keys():
                train_Xs[k].append(Xs[k][i])
            train_ys.append(ys[i])
    return train_Xs, train_ys, valid_Xs, valid_ys

def load_data(neighborhoods, feature_loaders):
    Xs = {k: list() for k in feature_loaders}
    ys = list()
    for n in neighborhoods:
        for coord, ls in landmarks.landmarks[n].items():
            y = [0.0]*len(landmarks.itos)
            for k, feature_loader in feature_loaders.items():
                Xs[k].append(feature_loader.get(n, coord[0], coord[1]))

            for l in ls:
                y[landmarks.stoi[l]] = 1.0
            ys.append(y)
            # print(ls, n, get_orientation_keys(coord[0], coord[1]))

    return Xs, ys

def batchify(X, ys, weights, cuda=True):
    bsz = len(ys)
    batch = dict()
    if 'resnet' in X:
        batch['resnet'] = torch.FloatTensor(X['resnet'])
    if 'fasttext' in X:
        max_len = max(len(s) for s in X['fasttext'])
        batch['fasttext'] = torch.FloatTensor(bsz, max_len, X['fasttext'][0][0].shape[0]).zero_()
        for ii in range(bsz):
            for jj in range(len(X['fasttext'][ii])):
                batch['fasttext'][ii, jj, :] = torch.from_numpy(X['fasttext'][ii][jj])
    if 'textrecog' in X:
        max_len = max(len(s) for s in X['textrecog'])
        batch['textrecog'] = torch.LongTensor(bsz, max_len).zero_()
        for ii in range(bsz):
            for jj in range(len(X['textrecog'][ii])):
                batch['textrecog'][ii][jj] = X['textrecog'][ii][jj]
    return to_variable((batch, torch.FloatTensor(ys), weights), cuda=cuda)


class LandmarkClassifier(nn.Module):

    def __init__(self, textrecog_features, fasttext_features, resnet_features, num_tokens=100, pool='sum', resnet_dim=2048, fasttext_dim=300):
        super().__init__()
        self.textrecog_features = textrecog_features
        self.fasttext_features = fasttext_features
        self.resnet_features = resnet_features
        self.pool = pool

        if self.fasttext_features:
            self.fasttext_linear = nn.Linear(fasttext_dim, 9, bias=False)
        if self.resnet_features:
            self.resnet_linear = nn.Linear(resnet_dim, 9, bias=False)
        if self.textrecog_features:
            self.embed = nn.Embedding(num_tokens, 50, padding_idx=0)
            self.textrecog_linear = nn.Linear(50, 9, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, X, y, weights):
        batchsize = y.size(0)
        logits = Variable(torch.FloatTensor(batchsize, 9)).zero_()

        if self.textrecog_features:
            embeddings = self.embed(X['textrecog'])
            for i in range(batchsize):
                if self.pool == 'sum':
                    logits[i, :] += self.textrecog_linear(embeddings[i, :, :]).sum(dim=0)
                else:
                    logits[i, :] += self.textrecog_linear(embeddings[i, :, :]).max(dim=0)[0]

        if self.fasttext_features:
            for i in range(batchsize):
                if self.pool == 'sum':
                    logits[i, :] += self.fasttext_linear(X['fasttext'][i, :, :]).sum(dim=0)
                else:
                    logits[i, :] += self.fasttext_linear(X['fasttext'][i, :, :]/rescale_factor).max(dim=0)[0]

        if self.resnet_features:
            for i in range(batchsize):
                if self.pool == 'sum':
                    logits[i, :] += self.resnet_linear(X['resnet'][i, :, :]).sum(dim=0)
                else:
                    logits[i, :] += self.resnet_linear(X['resnet'][i, :, :]).max(dim=0)[0]


        self.loss.weight = weights.view(-1).data

        target = y.view(-1)
        loss = self.loss(logits.view(-1), target)

        y_pred = torch.ge(self.sigmoid(logits), 0.5).float().data.numpy()
        y_true = y.data.numpy()

        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        return loss, f1, precision, recall

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--resnet-features', action='store_true')
    parser.add_argument('--textrecog-features', action='store_true')
    parser.add_argument('--fasttext-features', action='store_true')
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--n_components', type=int, default=100)
    parser.add_argument('--pool', choices=['max', 'sum'], default='sum')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--exp-name', type=str, default='test')

    args = parser.parse_args()
    torch.manual_seed(0)

    exp_dir = os.path.join(os.environ['TALKTHEWALK_EXPDIR'], args.exp_name)
    if os.path.exists(exp_dir):
        raise RuntimeError('Experiment directory already exist..')
    os.mkdir(exp_dir)

    logger = create_logger(os.path.join(exp_dir, 'log.txt'))
    logger.info(args)

    print(args)
    neighborhoods = ['fidi', 'hellskitchen', 'williamsburg', 'uppereast', 'eastvillage']
    landmarks = Map(neighborhoods)
    data_dir = './data'

    feature_loaders = dict()
    num_tokens = None
    if args.fasttext_features:
        textfeatures = load_features(neighborhoods)
        obs_i2s, obs_s2i = create_obs_dict(textfeatures, neighborhoods)
        feature_loaders['fasttext'] = FasttextFeatures(textfeatures, '/private/home/harm/data/wiki.en.bin', pca=args.pca, n_components=args.n_components)
    if args.resnet_features:
        feature_loaders['resnet'] = ResnetFeatures(os.path.join(data_dir, 'resnetfeat.json'), pca=args.pca, n_components=args.n_components)
    if args.textrecog_features:
        textfeatures = load_features(neighborhoods)
        obs_i2s, obs_s2i = create_obs_dict(textfeatures, neighborhoods)
        feature_loaders['textrecog'] = TextrecogFeatures(textfeatures, obs_s2i)
        num_tokens = len(obs_i2s)
    assert (len(feature_loaders) > 0)

    Xs, ys = load_data(neighborhoods, feature_loaders)

    train_Xs, train_ys, valid_Xs, valid_ys = create_split(Xs, ys)

    ys = torch.FloatTensor(train_ys)
    positives = ys.sum(0)

    train_weights = torch.FloatTensor(len(train_ys), len(landmarks.itos)).fill_(0.0)
    for i in range(len(train_ys)):
        for j in range(len(landmarks.itos)):
            if train_ys[i][j] == 1.:
                train_weights[i, j] = 1.0 /positives[j]
            else:
                train_weights[i, j] = 1.0 /(len(train_ys) - positives[j])

    valid_weights = torch.FloatTensor(len(valid_ys), len(landmarks.itos)).fill_(0.0)
    for i in range(len(valid_ys)):
        for j in range(len(landmarks.itos)):
            if valid_ys[i][j] == 1.:
                valid_weights[i, j] = 1. / positives[j]
            else:
                valid_weights[i, j] = 1. /(len(train_ys) - positives[j])

    resnet_dim, fasttext_dim = 2048, 300
    if args.pca:
        resnet_dim = args.n_components
        fasttext_dim = args.n_components
    net = LandmarkClassifier(args.textrecog_features, args.fasttext_features, args.resnet_features,
                             num_tokens=num_tokens, pool=args.pool, resnet_dim=resnet_dim, fasttext_dim=fasttext_dim)
    # if args.fasttext_features:
        # landmark_embeddings = torch.FloatTensor(9, 300)
        # for j in range(9):
        #     emb = feature_loaders['fasttext'].f.get_word_vector(landmarks.itos[j].lower().split(" ")[0])
        #     rescale_factor = numpy.linalg.norm(emb, 2) / net.fasttext_linear.weight.data[j, :].norm(2)
        #     landmark_embeddings[j, :] = torch.from_numpy(emb)/rescale_factor
        # net.fasttext_linear.weight.data = landmark_embeddings
    opt = optim.Adam(net.parameters())


    train_Xs, train_ys, train_weights = batchify(train_Xs, train_ys, train_weights, cuda=args.cuda)
    valid_Xs, valid_ys, valid_weights = batchify(valid_Xs, valid_ys, valid_weights, cuda=args.cuda)


    target = valid_ys.data.numpy()
    ones = numpy.ones_like(target)
    rand = numpy.random.randint(2, size=target.shape)

    print("All positive: {}, {}, {}".format(f1_score(target, ones, average='weighted'), precision_score(target, ones, average='weighted'), recall_score(target, ones, average='weighted')))
    print("Random (0.5): {}, {}, {}".format(f1_score(target, rand, average='weighted'), precision_score(target, rand, average='weighted'), recall_score(target, rand, average='weighted')))

    # NN classifier
    if args.fasttext_features or args.resnet_features:
        classifiers = list()
        k = list(train_Xs.keys())[0]
        for j in range(train_ys.size(1)):
            classifiers.append(KNeighborsClassifier(n_neighbors=1))
            train_feats = train_Xs[k].sum(dim=1).data.numpy()
            train_labels = train_ys[:, j].data.numpy()
            # data = train_Xs[k]
            classifiers[j].fit(train_feats, train_labels)

        nn_pred = numpy.zeros((valid_ys.size(0), len(classifiers)))
        valid_feats = valid_Xs[k][:, :, :].sum(dim=1)
        for j, classifier in enumerate(classifiers):
            nn_pred[:, j] = classifier.predict(valid_feats)

        print("NN: {}, {}, {}".format(f1_score(target, nn_pred, average='weighted'),
                                         precision_score(target, nn_pred, average='weighted'),
                                         recall_score(target, nn_pred, average='weighted')))

    train_f1s = list()
    test_f1s = list()
    train_losses = list()
    test_losses = list()

    best_val_loss = 1e10
    best_train_loss = 0.0
    best_train_f1 = 0.0
    best_val_f1 = 0.0
    best_val_precision = 0.0
    best_val_recall = 0.0

    for i in range(args.num_epochs):
        train_loss, train_f1, train_precision, train_recall = net.forward(train_Xs, train_ys, train_weights)
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        valid_loss, valid_f1, valid_precision, valid_recall = net.forward(valid_Xs, valid_ys, valid_weights)
        # valid_loss, valid_f1 = Variable(torch.FloatTensor([0.0])), 0.0

        logger.info("Train loss: {} | Train f1: {} | Train precision: {} | Train recall: {} |"
                    " Valid loss: {} | Valid f1: {} | Valid precision: {} | valid recall: {}".format(
            train_loss.data.numpy()[0],
            train_f1,
            train_precision,
            train_recall,
            valid_loss.data.numpy()[0],
            valid_f1,
            valid_precision,
            valid_recall))
        train_losses.append(train_loss.data.numpy()[0])
        test_losses.append(valid_loss.data.numpy()[0])
        train_f1s.append(train_f1)
        test_f1s.append(valid_f1)

        if valid_loss.data.numpy()[0] < best_val_loss:
            best_train_f1 = train_f1
            best_val_f1 = valid_f1
            best_val_loss = valid_loss.data.numpy()[0]
            best_train_loss = train_loss.data.numpy()[0]
            best_val_precision = valid_precision
            best_val_recall = valid_recall

    logger.info("{}, {}, {}, {}, {}, {}".format(best_train_loss, best_val_loss, best_train_f1, best_val_f1, best_val_precision, best_val_recall))

    plt.plot(range(len(train_losses)), train_losses, label='train')
    plt.plot(range(len(test_losses)), test_losses, label='test')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'loss.png'))
    plt.clf()

    plt.plot(range(len(train_f1s)), train_f1s, label='train')
    plt.plot(range(len(test_f1s)), test_f1s, label='test')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'f1.png'))
