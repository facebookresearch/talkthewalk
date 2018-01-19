import argparse
import json
import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.metrics import f1_score, precision_recall_fscore_support
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from data_loader import Landmarks, create_obs_dict, load_features, get_orientation_keys

neighborhoods = ['fidi', 'uppereast', 'eastvillage', 'williamsburg', 'hellskitchen']
landmarks = Landmarks(neighborhoods)

random.seed(0)
torch.manual_seed(0)

def create_split(Xs, ys):
    train_Xs = list()
    train_ys = list()
    valid_Xs = list()
    valid_ys = list()

    for data, label in zip(Xs, ys):
        if random.random() > 0.7:
            valid_Xs.append(data)
            valid_ys.append(label)
        else:
            train_Xs.append(data)
            train_ys.append(label)
    return train_Xs, train_ys, valid_Xs, valid_ys

def batchify(Xs, ys, weights, use_fasttext=False, features=None):
    if features == 'textrecog':
        token_lens = [len(x) for x in Xs]
        if use_fasttext:
            X = torch.FloatTensor(len(Xs), max(token_lens), 300).fill_(0.)
            for i, x in enumerate(Xs):
                if len(x) > 0:
                    X[i, :len(x), :] = torch.from_numpy(numpy.array(x))
        else:
            X = torch.LongTensor(len(Xs), max(token_lens)).fill_(0)
            for i, x in enumerate(Xs):
                if len(x) > 0:
                    X[i, :len(x)] = torch.LongTensor(x)
    else:
        X = torch.FloatTensor(Xs)

    X = Variable(X)
    y = Variable(torch.FloatTensor(ys))

    return X, y, weights

class TextrecogNetwork(nn.Module):

    def __init__(self, num_tokens, use_fasttext, pool):
        super().__init__()
        self.use_fasttext = use_fasttext
        self.pool = pool
        # self.linear2 = nn.Linear(4*2048, 9)
        # self.linear = nn.Linear(2048, 9)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss(size_average=True)

        if not use_fasttext:
            self.embed = nn.Embedding(num_tokens, 300, padding_idx=0)
        self.linear = nn.Linear(300, 9, bias=False)


    def forward(self, x, y, weights, c_weights):
        # out = self.linear2(x)
        # out = self.linear(F.relu(out))
        batchsize = x.size(0)
        if not self.use_fasttext:
            textrecog_embeddings = self.embed(x)
        else:
            textrecog_embeddings = x

        logits = Variable(torch.FloatTensor(batchsize, 9))
        for i in range(batchsize):
            if self.pool == 'sum':
                logits[i, :] = self.linear(textrecog_embeddings[i, :, :]).sum(dim=0)
            else:
                logits[i, :] = self.linear(textrecog_embeddings[i, :, :]).max(dim=0)[0]

        # NLL
        self.loss.weight = weights.view(-1)
        prob = self.sigmoid(logits)

        target = y.view(-1)
        loss = self.loss(prob.view(-1), target)

        y_pred = torch.ge(prob.float(), 0.5).float().data.numpy()
        y_true = y.data.numpy()

        f1 = f1_score(y_true, y_pred, average='weighted')

        return loss, f1

class ResnetNetwork(nn.Module):

    def __init__(self, pool):
        super().__init__()
        self.pool = pool
        self.linear = nn.Linear(2048, 9)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, y, weights):
        # out = self.linear2(x)
        # out = self.linear(F.relu(out))
        x = x*2.0
        batchsize = x.size(0)
        logits = Variable(torch.FloatTensor(batchsize, 9))
        for i in range(batchsize):
            if self.pool == 'sum':
                logits[i, :] = self.linear(x[i, :, :]).sum(dim=0)
            else:
                logits[i, :] = self.linear(x[i, :, :]).max(dim=0)[0]

        self.loss.weight = weights.view(-1)

        target = y.view(-1)
        loss = self.loss(logits.view(-1), target)

        y_pred = torch.ge(logits.float(), 0.0).float().data.numpy()
        y_true = y.data.numpy()

        f1 = f1_score(y_true, y_pred, average='weighted')

        return loss, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', choices=['textrecog', 'resnet'], default='textrecog')
    parser.add_argument('--use-fasttext', action='store_true')
    parser.add_argument('--pool', choices=['max', 'sum'], default='sum')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--exp-name', type=str, default='test')

    args = parser.parse_args()

    print(args)

    if args.use_fasttext:
        import fastText
        f = fastText.load_model('/private/home/harm/data/wiki.en.bin')

    if args.features == 'textrecog':
        text_features = load_features(neighborhoods)
        obs_i2s, obs_s2i = create_obs_dict(text_features, neighborhoods)
    else:
        resnet_features = json.load(open('resnetfeat.json'))

    Xs = []
    ys = []
    for n in neighborhoods:
        for coord, ls in landmarks.landmarks[n].items():
            y = [0.0]*len(landmarks.itos)
            x_features = list()
            for key in get_orientation_keys(coord[0], coord[1]):
                if args.features == 'textrecog':
                    for recog in text_features[n][key]:
                        if args.use_fasttext:
                            x_features.append(f.get_word_vector(recog['lex_recog']))
                        else:
                            x_features.append(obs_s2i[recog['lex_recog']])
                else:
                    #res net
                    x_features.append(resnet_features[n][key])
            Xs.append(x_features)
            for l in ls:
                y[landmarks.stoi[l]] = 1.0
            ys.append(y)

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

    if args.features == 'textrecog':
        net = TextrecogNetwork(len(obs_i2s), args.use_fasttext, args.pool)
        if args.use_fasttext:
            class_embedding = torch.FloatTensor(9, 300).zero_()
            for i in range(9):
                class_embedding[i, :] = torch.FloatTensor(f.get_word_vector(landmarks.itos[i].lower().split(" ")[0]))/100.
            net.linear.weight = nn.Parameter(class_embedding, requires_grad=True)
        for k, p in net.named_parameters():
            print(k)
        opt = optim.Adam(net.parameters())
    else:
        net = ResnetNetwork(args.pool)
        opt = optim.Adam(net.parameters())


    train_Xs, train_ys, train_weights = batchify(train_Xs, train_ys, train_weights, features=args.features, use_fasttext=args.use_fasttext)
    valid_Xs, valid_ys, valid_weights = batchify(valid_Xs, valid_ys, valid_weights, features=args.features, use_fasttext=args.use_fasttext)

    train_f1s = list()
    test_f1s = list()

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

        print("Train loss: {} | Train f1: {} | Valid loss: {} | Valid f1: {}".format(train_loss.data.numpy()[0],
                                                                                     train_f1,
                                                                                     valid_loss.data.numpy()[0],
                                                                                     valid_f1))
        train_f1s.append(train_loss.data.numpy()[0])
        test_f1s.append(valid_loss.data.numpy()[0])
        if valid_loss.data.numpy()[0] < best_val_loss:
            best_train_f1 = train_f1
            best_val_f1 = valid_f1
            best_val_loss = valid_loss.data.numpy()[0]
            best_train_loss = train_loss.data.numpy()[0]

    print(best_train_loss, best_val_loss, best_train_f1, best_val_f1)

    plt.plot(range(len(train_f1s)), train_f1s, label='train')
    plt.plot(range(len(test_f1s)), test_f1s, label='test')
    plt.legend()
    plt.savefig('{}.png'.format(args.exp_name))