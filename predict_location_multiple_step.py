import argparse
import json
import os
import time
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from functools import reduce
from data_loader import Landmarks, FasttextFeatures, ResnetFeatures, GoldstandardFeatures, \
                        load_data_multiple_step, load_features, create_obs_dict, to_variable
from utils import create_logger

class MapEmbedding2d(nn.Module):

    def __init__(self, num_tokens, emb_size, init_std=1.0):
        super(MapEmbedding2d, self).__init__()
        self.emb_landmark = nn.Embedding(num_tokens, emb_size, padding_idx=0)
        if init_std != 1.0:
            self.emb_landmark.weight.data.normal_(0.0, init_std)
        self.emb_size = emb_size

    def forward(self, x):
        shape = x.size()
        num_elements = reduce(lambda x, y: x * y, shape)
        flatten_x = x.resize(num_elements)
        emb_x = self.emb_landmark.forward(flatten_x)
        reshaped_emb = emb_x.resize(*shape, self.emb_size)
        return reshaped_emb.sum(dim=-2)

class LocationPredictor(nn.Module):

    def __init__(self, goldstandard_features, resnet_features, fasttext_features,
                 emb_sz, num_embeddings, max_steps=2, mask_conv=False, condition_on_action=False):
        super(LocationPredictor, self).__init__()
        self.goldstandard_features = goldstandard_features
        self.resnet_features = resnet_features
        self.fasttext_features = fasttext_features
        self.num_embeddings = num_embeddings
        self.emb_sz = emb_sz
        if self.goldstandard_features:
            self.goldstandard_emb = nn.Embedding(11, emb_sz)
        if self.fasttext_features:
            self.fasttext_emb_linear = nn.Linear(300, emb_sz)
        if self.resnet_features:
            self.resnet_emb_linear = nn.Linear(2048, emb_sz)
        self.emb_map = MapEmbedding2d(num_embeddings, emb_sz, init_std=0.01)
        self.max_steps = max_steps
        self.conv_weight = nn.Parameter(torch.FloatTensor(
                emb_sz, emb_sz, 3, 3).cuda())
        # weight = Variable(torch.FloatTensor(emb_sz, emb_sz, 3, 3)).cuda()
        # for i in range(3):
        #     for j in range(3):
        #         weight[:, :, i, j] = Variable(torch.eye(emb_sz, emb_sz))
        # self.conv_weight = weight
        std = 1.0/(emb_sz*9)
        self.conv_weight.data.uniform_(-std, std)
        self.loss = nn.CrossEntropyLoss()
        self.mask_conv = mask_conv
        self.condition_on_action = condition_on_action
        if self.condition_on_action:
            self.action_emb = nn.Embedding(4, 32)
            self.action_linear = nn.Linear(32, 9)

    def cuda(self):
        self.emb_map.cuda()
        self.goldstandard_emb.cuda()
        if self.condition_on_action:
            self.action_emb.cuda()
            self.action_linear.cuda()

    def forward(self, X, actions, landmarks, y):
        batch_size = y.size(0)
        if self.goldstandard_features:
            max_steps = X['goldstandard'].size(1)
            emb = list()
            for step in range(max_steps):
                emb.append(self.goldstandard_emb.forward(X['goldstandard'][:, step, :]).sum(dim=1))
            goldstandard_emb = torch.cat(emb, 1)

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

        l_emb = self.emb_map.forward(landmarks).permute(0, 3, 1, 2)
        l_embs = [l_emb]

        if self.condition_on_action:
            action_inp = Variable(torch.LongTensor([0, 1, 2, 3]).cuda())
            action_emb = self.action_emb.forward(action_inp)
            action_out = self.action_linear(action_emb)

            # action_out = Variable(torch.FloatTensor(4, 3, 3).fill_(0.0).cuda())
            # action_out[0, 1, 2] = 1.0
            # action_out[1, 1, 0] = 1.0
            # action_out[2, 2, 1] = 1.0
            # action_out[3, 0, 1] = 1.0

            for j in range(self.max_steps - 1):
                # SPEEDUP: iterate over actions (4) rather than batch size
                out = Variable(torch.FloatTensor(batch_size, self.emb_sz, 4, 4).zero_().cuda())
                for k in range(4):
                    if (actions[:, j] == k).nonzero().size(0) > 0:
                        selected_inp = l_embs[-1][(actions[:, j] == k), :, :, :]
                        mask = F.softmax(action_out[k], dim=0).resize(1, 1, 3, 3)
                        weight = mask*self.conv_weight
                        out[(actions[:, j] == k), :, :, :] = F.conv2d(selected_inp, weight, padding=1)
                l_embs.append(out)
        else:
            weight = self.conv_weight
            if self.mask_conv:
                mask = torch.FloatTensor(1, 1, 3, 3).cuda().zero_()
                mask[0, 0, 0, 1] = 1.0
                mask[0, 0, 1, 0] = 1.0
                mask[0, 0, 2, 1] = 1.0
                mask[0, 0, 1, 2] = 1.0
                weight = self.conv_weight * Variable(mask)
            for j in range(self.max_steps-1):
                tmp = F.conv2d(l_embs[-1], weight, padding=1)
                l_embs.append(tmp)

        neural_localisation = False
        cosine_similarity = False
        if neural_localisation:
            belief = Variable(torch.FloatTensor(batch_size, 16).fill_(1/16.0).cuda())
            for i in range(self.max_steps):
                landmark_embs = l_embs[i].resize(batch_size, self.emb_sz, 16).transpose(1, 2)
                likelihood = torch.bmm(landmark_embs, emb[:, i*self.emb_sz:(i+1)*self.emb_sz].unsqueeze(-1)).squeeze(-1)
                belief = F.softmax(likelihood*belief, dim=1)
            prob = belief
        elif cosine_similarity:
            logits = Variable(torch.FloatTensor(batch_size, 16).zero_().cuda())
            for i in range(self.max_steps):
                landmarks = l_embs[i].resize(batch_size, self.emb_sz, 16).transpose(1, 2)
                logits += batch_cosine(landmarks, emb[:, i*self.emb_sz:(i+1)*self.emb_sz])
            prob = F.softmax(logits, dim=1)
        else:
            landmarks = torch.cat(l_embs, 1)
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
        state['max_steps'] = self.max_steps
        state['mask_conv'] = self.mask_conv
        state['condition-on-action'] = self.condition_on_action
        state['parameters'] = self.state_dict()
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        model = cls(state['goldstandard_features'], state['resnet_features'], state['fasttext_features'],
                    state['emb_sz'], state['num_embeddings'], max_steps=state['max_steps'],
                    mask_conv=state['mask_conv'], condition_on_action=state['condition-on-action'])
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


def batch_cosine(A, B):
    batch_size = A.size(0)
    resized_A = A.resize(batch_size*A.size(1), A.size(2))
    expanded_B = B.unsqueeze(1).expand(batch_size, A.size(1), B.size(1))
    resized_B = expanded_B.resize(batch_size*A.size(1), B.size(1))
    return F.cosine_similarity(resized_A, resized_B, 1, 1e-6).resize(batch_size, A.size(1))


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

        l += loss.cpu().data.numpy()[0]
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
    parser.add_argument('--condition-on-action', action='store_true')
    parser.add_argument('--mask-conv', action='store_true')
    parser.add_argument('--num-steps', type=int, default=2)
    parser.add_argument('--softmax', choices=['landmarks', 'location'], default='location')
    parser.add_argument('--emb-sz', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--batch_sz', type=int, default=64)
    parser.add_argument('--exp-name', type=str, default='test')

    args = parser.parse_args()

    print(args)

    exp_dir = os.path.join(os.environ['TALKTHEWALK_EXPDIR'], args.exp_name)
    if os.path.exists(exp_dir):
        raise RuntimeError('Experiment directory already exist..')
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

    X_train, actions_train, landmark_train, y_train = load_data_multiple_step(train_configs, feature_loaders, landmark_map,
                                                                              softmax=args.softmax, num_steps=args.num_steps)
    X_valid, actions_valid, landmark_valid, y_valid = load_data_multiple_step(valid_configs, feature_loaders, landmark_map,
                                                                              softmax=args.softmax, num_steps=args.num_steps)
    X_test, actions_test, landmark_test, y_test = load_data_multiple_step(test_configs, feature_loaders, landmark_map,
                                                                          softmax=args.softmax, num_steps=args.num_steps)

    print(len(X_train['goldstandard']), len(X_valid['goldstandard']), len(X_test['goldstandard']))

    num_embeddings = len(landmark_map.idx_to_global_coord)
    if args.softmax == 'landmarks':
        num_embeddings = len(landmark_map.itos) + 1

    net = LocationPredictor(args.goldstandard_features, args.resnet_features, args.fasttext_features, args.emb_sz, num_embeddings,
                            condition_on_action=args.condition_on_action, mask_conv=args.mask_conv, max_steps=args.num_steps)
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

        sim_matrix = Variable(torch.FloatTensor(11, 11))
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
