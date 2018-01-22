import argparse
import json
import random
import numpy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from data_loader import Landmarks, create_obs_dict, load_features, ResnetFeatures, FasttextFeatures, TextrecogFeatures, to_variable
from utils import create_logger

neighborhoods = ['fidi', 'uppereast', 'eastvillage', 'williamsburg', 'hellskitchen']
landmarks = Landmarks(neighborhoods)

random.seed(3)
torch.manual_seed(0)

def create_split(Xs, ys):
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

    return Xs, ys

def batchify(X, ys, weights, cuda=True):
    bsz = len(ys)
    batch = dict()
    if 'resnet' in X:
        batch['resnet'] = torch.FloatTensor(X['resnet'])
    if 'fasttext' in X:
        max_len = max(len(s) for s in X['fasttext'])
        batch['fasttext'] = torch.FloatTensor(bsz, max_len, 300).zero_()
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

    def __init__(self, textrecog_features, fasttext_features, resnet_features, num_tokens=100, pool='sum'):
        super().__init__()
        self.textrecog_features = textrecog_features
        self.fasttext_features = fasttext_features
        self.resnet_features = resnet_features
        self.pool = pool

        if self.fasttext_features:
            self.fasttext_linear = nn.Linear(300, 9, bias=False)
        if self.resnet_features:
            self.resnet_linear = nn.Linear(2048, 9, bias=False)
        if self.textrecog_features:
            self.embed = nn.Embedding(num_tokens, 300, padding_idx=0)
            self.textrecog_linear = nn.Linear(300, 9, bias=False)
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
                rescale_factor = X['fasttext'][0, :, :].data.norm(2) / self.fasttext_linear.weight.data[0, :].norm(2)
                if self.pool == 'sum':
                    logits[i, :] += self.fasttext_linear(X['fasttext'][i, :, :]/rescale_factor).sum(dim=0)
                else:
                    logits[i, :] += self.fasttext_linear(X['fasttext'][i, :, :]/rescale_factor).max(dim=0)[0]

        if self.resnet_features:
            for i in range(batchsize):
                rescale_factor = X['resnet'][0, :, :].data.norm(2) / self.resnet_linear.weight.data[0, :].norm(2)
                if self.pool == 'sum':
                    logits[i, :] += self.resnet_linear(X['resnet'][i, :, :]/rescale_factor).sum(dim=0)
                else:
                    logits[i, :] += self.resnet_linear(X['resnet'][i, :, :]/rescale_factor).max(dim=0)[0]


        self.loss.weight = weights.view(-1).data

        target = y.view(-1)
        loss = self.loss(logits.view(-1), target)

        y_pred = torch.ge(logits.float(), 0.0).float().data.numpy()
        y_true = y.data.numpy()

        f1 = f1_score(y_true, y_pred, average='weighted')

        return loss, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--resnet-features', action='store_true')
    parser.add_argument('--textrecog-features', action='store_true')
    parser.add_argument('--fasttext-features', action='store_true')
    parser.add_argument('--pool', choices=['max', 'sum'], default='sum')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--exp-name', type=str, default='test')

    args = parser.parse_args()

    exp_dir = os.path.join(os.environ['TALKTHEWALK_EXPDIR'], args.exp_name)
    if os.path.exists(exp_dir):
        raise RuntimeError('Experiment directory already exist..')
    os.mkdir(exp_dir)

    logger = create_logger(os.path.join(exp_dir, 'log.txt'))
    logger.info(args)

    print(args)
    neighborhoods = ['fidi', 'hellskitchen', 'williamsburg', 'uppereast', 'eastvillage']
    landmarks = Landmarks(neighborhoods)
    data_dir = './data'

    feature_loaders = dict()
    num_tokens = None
    if args.fasttext_features:
        textfeatures = load_features(neighborhoods)
        obs_i2s, obs_s2i = create_obs_dict(textfeatures, neighborhoods)
        feature_loaders['fasttext'] = FasttextFeatures(textfeatures, '/private/home/harm/data/wiki.en.bin')
    if args.resnet_features:
        feature_loaders['resnet'] = ResnetFeatures(os.path.join(data_dir, 'resnetfeat.json'))
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


    net = LandmarkClassifier(args.textrecog_features, args.fasttext_features, args.resnet_features, num_tokens=num_tokens, pool=args.pool)
    if args.fasttext_features:
        landmark_embeddings = torch.FloatTensor(9, 300)
        for j in range(9):
            emb = feature_loaders['fasttext'].f.get_word_vector(landmarks.itos[j].lower().split(" ")[0])
            rescale_factor = numpy.linalg.norm(emb, 2) / net.fasttext_linear.weight.data[j, :].norm(2)
            landmark_embeddings[j, :] = torch.from_numpy(emb)/rescale_factor
        net.fasttext_linear.weight.data = landmark_embeddings
    opt = optim.Adam(net.parameters())


    train_Xs, train_ys, train_weights = batchify(train_Xs, train_ys, train_weights, cuda=args.cuda)
    valid_Xs, valid_ys, valid_weights = batchify(valid_Xs, valid_ys, valid_weights, cuda=args.cuda)

    train_f1s = list()
    test_f1s = list()
    train_losses = list()
    test_losses = list()

    best_val_loss = 1e10
    best_train_loss = 0.0
    best_train_f1 = 0.0
    best_val_f1 = 0.0


    for i in range(args.num_epochs):
        train_loss, train_f1 = net.forward(train_Xs, train_ys, train_weights)
        opt.zero_grad()
        train_loss.backward()
        opt.step()
        valid_loss, valid_f1 = net.forward(valid_Xs, valid_ys, valid_weights)
        # valid_loss, valid_f1 = Variable(torch.FloatTensor([0.0])), 0.0

        logger.info("Train loss: {} | Train f1: {} | Valid loss: {} | Valid f1: {}".format(train_loss.data.numpy()[0],
                                                                                     train_f1,
                                                                                     valid_loss.data.numpy()[0],
                                                                                     valid_f1))
        train_losses.append(train_loss.data.numpy()[0])
        test_losses.append(valid_loss.data.numpy()[0])
        train_f1s.append(train_f1)
        test_f1s.append(valid_f1)

        if valid_loss.data.numpy()[0] < best_val_loss:
            best_train_f1 = train_f1
            best_val_f1 = valid_f1
            best_val_loss = valid_loss.data.numpy()[0]
            best_train_loss = train_loss.data.numpy()[0]

    logger.info("{}, {}, {}, {}".format(best_train_loss, best_val_loss, best_train_f1, best_val_f1))

    plt.plot(range(len(train_losses)), train_losses, label='train')
    plt.plot(range(len(test_losses)), test_losses, label='test')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'loss.png'))
    plt.clf()

    plt.plot(range(len(train_f1s)), train_f1s, label='train')
    plt.plot(range(len(test_f1s)), test_f1s, label='test')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'f1.png'))