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
import torch.optim as optim

from torch.utils.data.dataloader import DataLoader

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

from ttw.models import LandmarkClassifier
from ttw.data_loader import TalkTheWalkLandmarks, DatasetHolder
from ttw.utils import get_collate_fn
from ttw.logger import create_logger


def create_split(dataset):
    keys = dataset[0].keys()
    train_data = {k: list() for k in keys}
    valid_data = {k: list() for k in keys}

    for i in range(len(dataset)):
        if random.random() > 0.7:
            for k in dataset[i].keys():
                valid_data[k].append(dataset[i][k])
        else:
            for k in dataset[i].keys():
                train_data[k].append(dataset[i][k])
    return train_data, valid_data


def add_weights(train_data, valid_data):
    """For landmark classifcation"""
    train_N = len(train_data['target'])
    train_tgts = numpy.array([train_data['target'][i] for i in range(train_N)])
    positives = train_tgts.sum(axis=0)

    train_data['weight'] = list()
    for i in range(train_N):
        weight = [0.0] * 10
        for j in range(10):
            if train_tgts[i][j] == 1:
                weight[j] = 1.0 / positives[j]
            else:
                weight[j] = 1.0 / (train_N - positives[j])
        train_data['weight'].append(weight)

    valid_data['weight'] = list()
    for i in range(len(valid_data['target'])):
        weight = [0.0] * 10
        for j in range(10):
            if valid_data['target'][i][j] == 1:
                weight[j] = 1.0 / positives[j]
            else:
                weight[j] = 1.0 / (train_N - positives[j])
        valid_data['weight'].append(weight)


def eval_epoch(loader, net, opt=None):
    loss, f1, precision, recall = 0.0, 0.0, 0.0, 0.0
    total = 0
    for batch in loader:
        batch_sz = batch['target'].size(0)
        out = net.forward(batch)
        loss += out['loss'].item() * batch_sz
        f1 += out['f1'].item() * batch_sz
        precision += out['precision'].item() * batch_sz
        recall += out['recall'].item() * batch_sz
        total += batch_sz

        if opt:
            opt.zero_grad()
            out['loss'].backward()
            opt.step()
    return loss / total, f1 / total, precision / total, recall / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to talkthewalk dataset')
    parser.add_argument('--exp-dir', type=str, default='./exp', help='Directory in which experiments will be stored')
    parser.add_argument('--exp-name', type=str, default='landmark_classification',
                        help='Name of the experiment. Results will be stored in args.exp_dir/args.exp_name')
    parser.add_argument('--cuda', action='store_true', help='If true, runs on gpu')
    parser.add_argument('--resnet-features', action='store_true', help='Use extracted resnet features?')
    parser.add_argument('--textrecog-features', action='store_true',
                        help='Use extracted text recognition featured from images?')
    parser.add_argument('--fasttext-features', action='store_true',
                        help='Use pre-trained FastText vectors for text recognition?')
    parser.add_argument('--nearest-neighbor', action='store_true',
                        help='If true, use a 1 nearest neighbour classifier over the features')
    parser.add_argument('--pca', action='store_true',
                        help='If true, reduce dimensionality of features (only applicable to resnet '
                             'and fasttext features)')
    parser.add_argument('--batch-sz', type=int, default=256, help='Batch size')
    parser.add_argument('--n_components', type=int, default=100,
                        help='The number of principal components to keep (only applicable when args.pca is true)')
    parser.add_argument('--pool', choices=['max', 'sum'], default='sum',
                        help='Whether to use sum or max pooling over the features from different views.')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')

    args = parser.parse_args()
    torch.manual_seed(0)

    exp_dir = os.path.join(args.exp_dir, args.exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    logger = create_logger(os.path.join(exp_dir, 'log.txt'))
    logger.info(args)

    data = TalkTheWalkLandmarks(args.data_dir, args.resnet_features, args.fasttext_features, args.textrecog_features)

    train_data, valid_data = create_split(data)
    add_weights(train_data, valid_data)
    train_data = DatasetHolder(train_data)
    valid_data = DatasetHolder(valid_data)

    train_loader = DataLoader(train_data, args.batch_sz, collate_fn=get_collate_fn(args.cuda), shuffle=True)
    valid_loader = DataLoader(valid_data, args.batch_sz, collate_fn=get_collate_fn(args.cuda))

    target = numpy.array([valid_data[i]['target'] for i in range(len(valid_data))])
    ones = numpy.ones_like(target)
    rand = numpy.random.randint(2, size=target.shape)

    logger.info('Baselines' + '-' * 70)
    logger.info("All positive: {}, {}, {}".format(f1_score(target, ones, average='weighted'),
                                                  precision_score(target, ones, average='weighted'),
                                                  recall_score(target, ones, average='weighted')))
    logger.info("Random (0.5): {}, {}, {}".format(f1_score(target, rand, average='weighted'),
                                                  precision_score(target, rand, average='weighted'),
                                                  recall_score(target, rand, average='weighted')))
    logger.info('-' * 80)

    # NN classifier
    if args.nearest_neighbor:
        classifiers = list()
        collate_fn = get_collate_fn(cuda=False)
        train_batch = collate_fn([train_data[i] for i in range(len(train_data))])
        valid_batch = collate_fn([valid_data[i] for i in range(len(valid_data))])
        k = list(train_batch.keys())[0]
        for j in range(10):
            classifiers.append(KNeighborsClassifier(n_neighbors=1))
            train_feats = train_batch[k].sum(dim=1).data.numpy()
            train_labels = train_batch['target'][:, j].data.numpy()
            classifiers[j].fit(train_feats, train_labels)

        nn_pred = numpy.zeros((len(valid_data), len(classifiers)))
        valid_feats = valid_batch[k].sum(dim=1)
        for j, classifier in enumerate(classifiers):
            nn_pred[:, j] = classifier.predict(valid_feats)

        logger.info("NN: {}, {}, {}".format(f1_score(target, nn_pred, average='weighted'),
                                            precision_score(target, nn_pred, average='weighted'),
                                            recall_score(target, nn_pred, average='weighted')))
    else:
        resnet_dim, fasttext_dim = 2048, 300
        if args.pca:
            resnet_dim = args.n_components
            fasttext_dim = args.n_components
        num_tokens = len(data.textrecog_dict) if args.textrecog_features or args.fasttext_features else None
        net = LandmarkClassifier(args.textrecog_features, args.fasttext_features, args.resnet_features,
                                 num_tokens=num_tokens, pool=args.pool, resnet_dim=resnet_dim,
                                 fasttext_dim=fasttext_dim)

        opt = optim.Adam(net.parameters())

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
            train_loss, train_f1, train_precision, train_recall = eval_epoch(train_loader, net, opt=opt)
            valid_loss, valid_f1, valid_precision, valid_recall = eval_epoch(valid_loader, net)

            logger.info("Train loss: {} | Train precision: {} | Train recall: {} |"
                        " Valid loss: {} | Valid precision: {} | valid recall: {}".format(
                train_loss,
                train_precision,
                train_recall,
                valid_loss,
                valid_precision,
                valid_recall))
            train_losses.append(train_loss)
            test_losses.append(valid_loss)
            train_f1s.append(train_f1)
            test_f1s.append(valid_f1)

            if valid_loss < best_val_loss:
                best_train_f1 = train_f1
                best_val_f1 = valid_f1
                best_val_loss = valid_loss
                best_train_loss = train_loss
                best_val_precision = valid_precision
                best_val_recall = valid_recall

        logger.info("{}, {}, {}, {}, {}, {}".format(best_train_loss, best_val_loss, best_train_f1, best_val_f1,
                                                    best_val_precision, best_val_recall))
