import argparse
import json
import os
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data_loader import Landmarks, FasttextFeatures, ResnetFeatures, GoldstandardFeatures, \
                        load_data, load_features, create_obs_dict, to_variable
from utils import create_logger
from modules import CBoW, MASC

class LocationPredictor(nn.Module):

    def __init__(self, goldstandard_features, resnet_features, fasttext_features,
                 emb_sz, num_embeddings, T=2, apply_masc=True):
        super(LocationPredictor, self).__init__()
        self.goldstandard_features = goldstandard_features
        self.resnet_features = resnet_features
        self.fasttext_features = fasttext_features
        self.num_embeddings = num_embeddings
        self.emb_sz = emb_sz
        self.apply_masc = apply_masc
        self.T = T
        if self.goldstandard_features:
            self.goldstandard_emb = nn.Embedding(11, emb_sz)
        if self.fasttext_features:
            self.fasttext_emb_linear = nn.Linear(300, emb_sz)
        if self.resnet_features:
            self.resnet_emb_linear = nn.Linear(2048, emb_sz)
        self.cbow_fn = CBoW(num_embeddings, emb_sz, init_std=0.01)

        self.masc_fn = MASC(emb_sz, apply_masc=apply_masc)
        self.loss = nn.CrossEntropyLoss()

        self.obs_write_gate = nn.ParameterList()
        self.landmark_write_gate = nn.ParameterList()
        for _ in range(T+1):
            self.obs_write_gate.append(nn.Parameter(torch.FloatTensor(1, emb_sz).normal_(0.0, 0.1)))
            self.landmark_write_gate.append(nn.Parameter(torch.FloatTensor(1, emb_sz, 1, 1).normal_(0.0, 0.1)))

        if self.apply_masc:
            self.action_emb = nn.Embedding(4, emb_sz)
            self.action_mask = nn.Parameter(torch.FloatTensor(1, T, emb_sz).normal_(0.0, 0.1))
            self.extract_fns = nn.ModuleList()
            for _ in range(self.T):
                self.extract_fns.append(nn.Linear(emb_sz, 9))

    def forward(self, X, actions, landmarks, y):
        batch_size = y.size(0)
        if self.goldstandard_features:
            max_steps = X['goldstandard'].size(1)
            embs = list()
            for step in range(max_steps):
                emb = self.goldstandard_emb.forward(X['goldstandard'][:, step, :]).sum(dim=1)
                emb = emb * F.sigmoid(self.obs_write_gate[step])
                embs.append(emb)
            goldstandard_emb = sum(embs)

        if self.resnet_features:
            resnet_emb = self.resnet_emb_linear.forward(X['resnet'])
            resnet_emb = resnet_emb.sum(dim=1)

        if self.fasttext_features:
            fasttext_emb = self.fasttext_emb_linear.forward(X['fasttext'])
            fasttext_emb = fasttext_emb.sum(dim=1)

        if self.resnet_features and self.fasttext_features:
            emb = resnet_emb + fasttext_emb
        elif self.resnet_features:
            emb = resnet_emb
        elif self.goldstandard_features:
            emb = goldstandard_emb
        elif self.fasttext_features:
            emb = fasttext_emb

        l_emb = self.cbow_fn.forward(landmarks).permute(0, 3, 1, 2)
        l_embs = [l_emb]

        if self.apply_masc:
            action_emb = self.action_emb.forward(actions)
            action_emb *= self.action_mask
            action_out = action_emb.sum(dim=1)

            for j in range(self.T):
                act_mask = self.extract_fns[j](action_out)
                out = self.masc_fn.forward(l_embs[-1], act_mask)
                l_embs.append(out)
        else:
            for j in range(self.T):
                out = self.masc_fn.forward_no_masc(l_emb)
                l_embs.append(out)

        landmarks = sum([F.sigmoid(gate)*emb for gate, emb in zip(self.landmark_write_gate, l_embs)])
        landmarks = landmarks.resize(batch_size, landmarks.size(1), 16).transpose(1, 2)

        logits = torch.bmm(landmarks, emb.unsqueeze(-1)).squeeze(-1)
        prob = F.softmax(logits, dim=1)

        y_true = (y[:, 0]*4 + y[:, 1]).squeeze()
        loss = self.loss(prob, y_true)
        acc = sum([1.0 for pred, target in zip(prob.max(1)[1].data.cpu().numpy(), y_true.data.cpu().numpy()) if pred == target])/batch_size
        return loss, acc, prob

    def save(self, path):
        state = dict()
        state['goldstandard_features'] = self.goldstandard_features
        state['resnet_features'] = self.resnet_features
        state['fasttext_features'] = self.fasttext_features
        state['emb_sz'] = self.emb_sz
        state['num_embeddings'] = self.num_embeddings
        state['T'] = self.T
        state['apply_masc'] = self.apply_masc
        state['parameters'] = self.state_dict()
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        model = cls(state['goldstandard_features'], state['resnet_features'], state['fasttext_features'],
                    state['emb_sz'], state['num_embeddings'], T=state['T'],
                    apply_masc=state['apply_masc'])
        model.load_state_dict(state['parameters'])
        return model

def create_batch(X, actions, landmarks, y, cuda=False):
    bsz = len(y)
    batch = dict()
    if 'resnet' in X:
        batch['resnet'] = torch.FloatTensor(X['resnet'])
    if 'fasttext' in X:
        max_len = max(len(s) for s in X['fasttext'])
        batch['fasttext'] = torch.FloatTensor(bsz, max_len, 100).zero_()
        for ii in range(bsz):
            for jj in range(len(X['fasttext'][ii])):
                batch['fasttext'][ii, jj, :] = torch.from_numpy(X['fasttext'][ii][jj])
    if 'textrecog' in X:
        max_len = max(len(s) for s in X['textrecog'])
        batch['textrecog'] = torch.LongTensor(bsz, max_len).zero_()
        for ii in range(bsz):
            for jj in range(len(X['textrecog'][ii])):
                batch['textrecog'][ii, jj] = X['textrecog'][ii][jj]
    if 'goldstandard' in X:
        max_steps = max(len(s) for s in X['goldstandard'])
        max_len = max([max([len(x) for x in l]) for l in X['goldstandard']])
        batch['goldstandard'] = torch.LongTensor(bsz, max_steps, max_len).zero_()
        for ii in range(bsz):
            for jj in range(len(X['goldstandard'][ii])):
                for kk in range(len(X['goldstandard'][ii][jj])):
                    batch['goldstandard'][ii, jj, kk] = X['goldstandard'][ii][jj][kk]

    max_landmarks_per_coord = max([max([max([len(y) for y in x]) for x in l]) for l in landmarks])
    landmark_batch = torch.LongTensor(bsz, 4, 4, max_landmarks_per_coord).zero_()

    for i, ls in enumerate(landmarks):
        for j in range(4):
            for k in range(4):
                landmark_batch[i, j, k, :len(landmarks[i][j][k])] = torch.LongTensor(landmarks[i][j][k])

    return to_variable((batch, torch.LongTensor(actions), landmark_batch, torch.LongTensor(y).unsqueeze(-1)), cuda=cuda)


def epoch(net, X, actions, landmarks, y, batch_sz, opt=None, shuffle=True, cuda=False):
    indices = [i for i in range(len(y))]
    if shuffle:
        numpy.random.shuffle(indices)

    l, a = 0.0, 0.0
    n_batches = 0
    for jj in range(0, len(y), batch_sz):
        slice = indices[jj:jj+batch_sz]
        X_batch = {k: [X[k][i] for i in slice] for k in X.keys()}
        action_batch = [actions[i] for i in slice]
        landmark_batch = [landmarks[i] for i in slice]
        y_batch = [y[i] for i in slice]

        X_batch, action_batch, landmark_batch, y_batch = create_batch(X_batch, action_batch, landmark_batch, y_batch,
                                                                      cuda=cuda)
        loss, acc, _ = net.forward(X_batch, action_batch, landmark_batch, y_batch)

        l += loss.cpu().data.numpy()
        a += acc
        n_batches += 1

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
    return l/n_batches, a/n_batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--resnet-features', action='store_true')
    parser.add_argument('--fasttext-features', action='store_true')
    parser.add_argument('--goldstandard-features', action='store_true')
    parser.add_argument('--masc', action='store_true')
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--softmax', choices=['landmarks', 'location'], default='landmarks')
    parser.add_argument('--emb-sz', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--batch_sz', type=int, default=64)
    parser.add_argument('--exp-name', type=str, default='test')

    args = parser.parse_args()

    print(args)

    exp_dir = os.path.join(os.environ['TALKTHEWALK_EXPDIR'], args.exp_name)
    if not os.path.exists(exp_dir):
        # raise RuntimeError('Experiment directory already exist..')
        os.mkdir(exp_dir)

    logger = create_logger(os.path.join(exp_dir, 'log.txt'))
    logger.info(args)

    data_dir = os.environ.get('TALKTHEWALK_DATADIR', './data')

    train_configs = json.load(open(os.path.join(data_dir, 'configurations.train.json')))
    valid_configs = json.load(open(os.path.join(data_dir, 'configurations.valid.json')))
    test_configs = json.load(open(os.path.join(data_dir, 'configurations.test.json')))

    numpy.random.shuffle(train_configs)

    neighborhoods = ['fidi', 'hellskitchen', 'williamsburg', 'uppereast', 'eastvillage']
    landmark_map = Landmarks(neighborhoods, include_empty_corners=True)

    feature_loaders = dict()
    if args.fasttext_features:
        textfeatures = load_features(neighborhoods)
        obs_i2s, obs_s2i = create_obs_dict(textfeatures, neighborhoods)
        feature_loaders['fasttext'] = FasttextFeatures(textfeatures, '/private/home/harm/data/wiki.en.bin')
    if args.resnet_features:
        feature_loaders['resnet'] = ResnetFeatures(os.path.join(data_dir, 'resnetfeat.json'))
    if args.goldstandard_features:
        feature_loaders['goldstandard'] = GoldstandardFeatures(landmark_map)
    assert (len(feature_loaders) > 0)

    X_train, actions_train, landmark_train, y_train = load_data(train_configs, feature_loaders, landmark_map,
                                                                softmax=args.softmax, num_steps=args.T+1)
    X_valid, actions_valid, landmark_valid, y_valid = load_data(valid_configs, feature_loaders, landmark_map,
                                                                softmax=args.softmax, num_steps=args.T+1)
    X_test, actions_test, landmark_test, y_test = load_data(test_configs, feature_loaders, landmark_map,
                                                            softmax=args.softmax, num_steps=args.T+1)

    print(len(X_train['goldstandard']), len(X_valid['goldstandard']), len(X_test['goldstandard']))

    num_embeddings = len(landmark_map.idx_to_global_coord)
    if args.softmax == 'landmarks':
        num_embeddings = len(landmark_map.itos) + 1

    net = LocationPredictor(args.goldstandard_features, args.resnet_features, args.fasttext_features, args.emb_sz, num_embeddings,
                            apply_masc=args.masc, T=args.T)

    # net = LocationPredictor.load('/private/home/harm/exp/bla_2/model.pt')
    params = [v for k, v in net.named_parameters()]

    opt = optim.Adam(params, lr=1e-4)

    if args.cuda:
        net.cuda()

    best_train_loss, best_valid_loss, best_test_loss = None, 1e16, None
    best_train_acc, best_valid_acc, best_test_acc = None, None, None

    for i in range(args.num_epochs):
        # train
        train_loss, train_acc = epoch(net, X_train, actions_train, landmark_train, y_train, args.batch_sz, opt=opt, cuda=args.cuda)
        valid_loss, valid_acc = epoch(net, X_valid, actions_valid, landmark_valid, y_valid, args.batch_sz, cuda=args.cuda)
        test_loss, test_acc = epoch(net, X_test, actions_test, landmark_test, y_test, args.batch_sz, cuda=args.cuda)

        logger.info("Train loss: {} | Valid loss: {} | Test loss: {}".format(train_loss,
                                                                       valid_loss,
                                                                       test_loss))
        logger.info("Train acc: {} | Valid acc: {} | Test acc: {}".format(train_acc,
                                                                    valid_acc,
                                                                    test_acc))

        # sim_matrix = Variable(torch.FloatTensor(11, 11))
        # for k in range(11):
        #     for l in range(11):
        #         sim_matrix[k, l] = torch.dot(net.goldstandard_emb.weight[k, :], net.emb_map.emb_landmark.weight[l, :])
        # print(sim_matrix)
        # print([s for s in landmark_map.itos])

        if valid_loss < best_valid_loss:
            best_train_loss = train_loss
            best_valid_loss = valid_loss
            best_train_loss = test_loss

            best_train_acc, best_valid_acc, best_test_acc = train_acc, valid_acc, test_acc

            net.save(os.path.join(exp_dir, 'model.pt'))

    logger.info("%.2f, %.2f. %.2f" % (best_train_acc*100, best_valid_acc*100, best_test_acc*100))
