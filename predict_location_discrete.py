from __future__ import division

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from data_loader import Landmarks, load_data, load_features, create_obs_dict, FasttextFeatures, GoldstandardFeatures, ResnetFeatures
from utils import create_logger
from predict_location_continuous import create_batch
from modules import MASC, NoMASC, CBoW


def eval_epoch(X, actions, landmarks, y, tourist, guide, batch_sz, cuda, t_opt=None, g_opt=None):
    tourist.eval()
    guide.eval()

    correct, total = 0, 0
    for ii in range(0, len(y), args.batch_sz):
        X_batch = {k: X[k][ii:ii + batch_sz] for k in X.keys()}
        actions_batch = actions[ii:ii + batch_sz]
        landmark_batch = landmarks[ii:ii + batch_sz]
        y_batch = y[ii:ii + args.batch_sz]
        batch_in, batch_actions, batch_landmarks, batch_tgt = create_batch(X_batch, actions_batch, landmark_batch,
                                                                           y_batch, cuda=args.cuda)
        # forward
        t_comms, t_probs, t_val = tourist(batch_in, batch_actions)
        if cuda:
            t_comms = [x.cuda() for x in t_comms]
        out_g = guide(t_comms, batch_landmarks)

        # acc
        tgt = (batch_tgt[:, 0]*4 + batch_tgt[:, 1])
        pred = torch.max(out_g, 1)[1]
        correct += sum(
            [1.0 for y_hat, y_true in zip(pred, tgt) if y_hat == y_true])
        total += len(y_batch)


        if t_opt and g_opt:
            # train if optimizers are specified
            g_loss = -torch.log(torch.gather(out_g, 1, tgt))
            _, max_ind = torch.max(out_g, 1)

            # tourist loss
            rewards = -g_loss  # tourist reward is log likelihood of correct answer

            t_rl_loss = 0.
            eps = 1e-16

            advantage = Variable((rewards.data - t_val.data))
            if args.cuda:
                advantage = advantage.cuda()
            t_val_loss = ((t_val - Variable(rewards.data)) ** 2).mean()  # mse

            for action, prob in zip(t_comms, t_probs):
                if args.cuda:
                    action = action.cuda()
                    prob = prob.cuda()
                action_prob = action * prob + (1.0 - action) * (1.0 - prob)

                t_rl_loss -= (torch.log(action_prob + eps) * advantage).sum()

            # backward
            g_opt.zero_grad()
            t_opt.zero_grad()
            g_loss.mean().backward()
            (t_rl_loss + t_val_loss).backward()
            torch.nn.utils.clip_grad_norm(tourist.parameters(), 5)
            torch.nn.utils.clip_grad_norm(guide.parameters(), 5)
            g_opt.step()
            t_opt.step()

    return correct/total


class Guide(nn.Module):

    def __init__(self, in_vocab_sz, num_landmarks, apply_masc=True, T=2):
        super(Guide, self).__init__()
        self.in_vocab_sz = in_vocab_sz
        self.num_landmarks = num_landmarks
        self.T = T
        self.apply_masc = apply_masc
        self.emb_map = CBoW(num_landmarks, in_vocab_sz, init_std=0.1)
        self.obs_emb_fn = nn.Linear(in_vocab_sz, in_vocab_sz)
        self.landmark_write_gate = nn.ParameterList()
        for _ in range(T+1):
            self.landmark_write_gate.append(nn.Parameter(torch.FloatTensor(1, in_vocab_sz, 1, 1).normal_(0.0, 0.1)))

        if apply_masc:
            self.masc_fn = MASC(in_vocab_sz)
        else:
            self.masc_fn = NoMASC(in_vocab_sz)
        if self.apply_masc:
            self.action_emb = nn.ModuleList()
            for i in range(T):
                self.action_emb.append(nn.Linear(in_vocab_sz, 9))

        self.act_lin = nn.Linear(in_vocab_sz, 9)


    def forward(self, message, landmarks):
        msg_obs = self.obs_emb_fn(message[0])
        batch_size = message[0].size(0)

        landmark_emb = self.emb_map.forward(landmarks).permute(0, 3, 1, 2)
        landmark_embs = [landmark_emb]

        if self.apply_masc:
            for j in range(self.T):
                act_msg = message[1]
                action_out = self.action_emb[j](act_msg)

                out = self.masc_fn.forward(landmark_embs[-1], action_out)
                landmark_embs.append(out)
        else:
            for j in range(self.T):
                out = self.masc_fn.forward(landmark_embs[-1])
                landmark_embs.append(out)

        landmarks = sum([F.sigmoid(gate)*emb for gate, emb in zip(self.landmark_write_gate, landmark_embs)])
        landmarks = landmarks.view(batch_size, landmarks.size(1), 16).transpose(1, 2)

        logits = torch.bmm(landmarks, msg_obs.unsqueeze(-1)).squeeze(-1)
        prob = F.softmax(logits, dim=1)
        return prob


    def save(self, path):
        state = dict()
        state['in_vocab_sz'] = self.in_vocab_sz
        state['num_landmarks'] = self.num_landmarks
        state['parameters'] = self.state_dict()
        state['T'] = self.T
        state['apply_masc'] = self.apply_masc
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        guide = cls(state['in_vocab_sz'], state['num_landmarks'], T=state['T'],
                    apply_masc=state['apply_masc'])
        guide.load_state_dict(state['parameters'])
        return guide


class Tourist(nn.Module):
    def __init__(self, goldstandard_features, resnet_features, fasttext_features,
                 vocab_sz, T=2, apply_masc=False):
        super(Tourist, self).__init__()
        self.goldstandard_features = goldstandard_features
        self.resnet_features = resnet_features
        self.fasttext_features = fasttext_features
        self.T = T
        self.apply_masc = apply_masc
        self.vocab_sz = vocab_sz

        if self.goldstandard_features:
            self.goldstandard_emb = nn.Embedding(11, vocab_sz)
        if self.fasttext_features:
            self.fasttext_emb_linear = nn.Linear(300, vocab_sz)
        if self.resnet_features:
            self.resnet_emb_linear = nn.Linear(2048, vocab_sz)

        self.num_embeddings = T+1
        self.obs_write_gate = nn.ParameterList()
        for _ in range(T+1):
            self.obs_write_gate.append(nn.Parameter(torch.FloatTensor(1, vocab_sz).normal_(0.0, 0.1)))

        if self.apply_masc:
            self.action_emb = nn.Embedding(4, vocab_sz)
            self.num_embeddings += T
            self.act_write_gate = nn.ParameterList()
            for _ in range(T):
                self.act_write_gate.append(nn.Parameter(torch.FloatTensor(1, vocab_sz).normal_(0.0, 0.1)))

        self.loss = nn.CrossEntropyLoss()
        self.value_pred = nn.Linear((1+int(self.apply_masc))*self.vocab_sz, 1)

    def forward(self, X, actions, greedy=False):
        batch_size = actions.size(0)
        feat_emb = list()
        if self.goldstandard_features:
            max_steps = X['goldstandard'].size(1)
            for step in range(max_steps):
                emb = self.goldstandard_emb.forward(X['goldstandard'][:, step, :]).sum(dim=1)
                emb = emb * F.sigmoid(self.obs_write_gate[step])
                feat_emb.append(emb)

        act_emb = list()
        if self.apply_masc:
            for step in range(self.T):
                emb = self.action_emb.forward(actions[:, step])
                emb = emb * F.sigmoid(self.act_write_gate[step])
                act_emb.append(emb)

        comms = list()
        probs = list()

        feat_embeddings = sum(feat_emb)
        feat_logits = feat_embeddings
        feat_prob = F.sigmoid(feat_logits).cpu()
        feat_msg = feat_prob.bernoulli().detach()

        probs.append(feat_prob)
        comms.append(feat_msg)

        if self.apply_masc:
            act_embeddings = sum(act_emb)
            act_logits = act_embeddings
            act_prob = F.sigmoid(act_logits).cpu()
            act_msg = act_prob.bernoulli().detach()

            probs.append(act_prob)
            comms.append(act_msg)

        if self.apply_masc:
            embeddings = torch.cat([feat_embeddings, act_embeddings], 1).resize(batch_size, 2*self.vocab_sz)
        else:
            embeddings = feat_embeddings
        value = self.value_pred(embeddings)

        return comms, probs, value

    def save(self, path):
        state = dict()
        state['goldstandard_features'] = self.goldstandard_features
        state['resnet_features'] = self.resnet_features
        state['fasttext_features'] = self.fasttext_features
        state['vocab_sz'] = self.vocab_sz
        state['T'] = self.T
        state['apply_masc'] = self.apply_masc
        state['parameters'] = self.state_dict()
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        tourist = cls(state['goldstandard_features'], state['resnet_features'], state['fasttext_features'],
                      state['vocab_sz'], T=state['T'], apply_masc=state['apply_masc'])
        tourist.load_state_dict(state['parameters'])

        return tourist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--resnet-features', action='store_true')
    parser.add_argument('--fasttext-features', action='store_true')
    parser.add_argument('--goldstandard-features', action='store_true')
    parser.add_argument('--softmax', choices=['landmarks', 'location'], default='landmarks')
    parser.add_argument('--masc', action='store_true')
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--vocab-sz', type=int, default=500)
    parser.add_argument('--batch-sz', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--exp-name', type=str, default='test')

    args = parser.parse_args()

    report_every = 1 # 10 epochs

    exp_dir = os.path.join(os.environ['TALKTHEWALK_EXPDIR'], args.exp_name)
    if not os.path.exists(exp_dir):
        # raise RuntimeError('Experiment directory already exist..')
        os.mkdir(exp_dir)

    logger = create_logger(os.path.join(exp_dir, 'log.txt'))
    logger.info(args)

    # Load data
    neighborhoods = ['fidi', 'hellskitchen', 'williamsburg', 'uppereast', 'eastvillage']
    landmark_map = Landmarks(neighborhoods, include_empty_corners=True)
    textfeatures = load_features(neighborhoods)

    data_dir = os.environ.get('TALKTHEWALK_DATADIR', './data')

    obs_i2s, obs_s2i = create_obs_dict(textfeatures, neighborhoods)

    train_configs = json.load(open(os.path.join(data_dir, 'configurations.train.json')))
    valid_configs = json.load(open(os.path.join(data_dir, 'configurations.valid.json')))
    test_configs = json.load(open(os.path.join(data_dir, 'configurations.test.json')))

    feature_loaders = dict()
    in_vocab_sz = None
    if args.fasttext_features:
        textfeatures = load_features(neighborhoods)
        obs_i2s, obs_s2i = create_obs_dict(textfeatures, neighborhoods)
        feature_loaders['fasttext'] = FasttextFeatures(textfeatures, '/private/home/harm/data/wiki.en.bin')
    if args.resnet_features:
        feature_loaders['resnet'] = ResnetFeatures(os.path.join(data_dir, 'resnetfeat.json'))
    if args.goldstandard_features:
        feature_loaders['goldstandard'] = GoldstandardFeatures(landmark_map)
        in_vocab_sz = len(landmark_map.landmark2i) + 1
    assert (len(feature_loaders) > 0)

    X_train, actions_train, landmark_train, y_train = load_data(train_configs, feature_loaders,
                                                                              landmark_map,
                                                                              softmax=args.softmax,
                                                                              num_steps=args.T+1)
    X_valid, actions_valid, landmark_valid, y_valid = load_data(valid_configs, feature_loaders,
                                                                landmark_map,
                                                                softmax=args.softmax,
                                                                num_steps=args.T+1)
    X_test, actions_test, landmark_test, y_test = load_data(test_configs, feature_loaders, landmark_map,
                                                            softmax=args.softmax,
                                                            num_steps=args.T+1)

    num_embeddings = len(landmark_map.landmark2i)+1
    if args.softmax == 'location':
        num_embeddings = 12000

    guide = Guide(args.vocab_sz, num_embeddings,
                  apply_masc=args.masc, T=args.T)
    tourist = Tourist(args.goldstandard_features, args.resnet_features, args.fasttext_features, args.vocab_sz,
                      apply_masc=args.masc, T=args.T)

    # guide = Guide.load('/u/devries/Documents/talkthewalk/results/disc_masc_3/guide.pt')
    # tourist = Tourist.load('/u/devries/Documents/talkthewalk/results/disc_masc_3/tourist.pt')

    if args.cuda:
        guide = guide.cuda()
        tourist = tourist.cuda()

    g_opt, t_opt = optim.Adam(guide.parameters()), optim.Adam(tourist.parameters())

    train_acc = list()
    val_acc = list()
    test_acc = list()

    best_train_acc, best_val_acc, best_test_acc = 0.0, 0.0, 0.0

    for epoch in range(1, args.num_epochs):
        # train
        # tourist.train(); guide.train()
        X_train['goldstandard'], actions_train, landmark_train, y_train = shuffle(X_train['goldstandard'], actions_train, landmark_train, y_train)

        train_accuracy = eval_epoch(X_train, actions_train, landmark_train, y_train,
                               tourist, guide, args.batch_sz, args.cuda,
                               t_opt=t_opt, g_opt=g_opt)
        # train_accuracy = eval_epoch(X_train, actions_train, landmark_train, y_train,
        #                        tourist, guide, args.batch_sz, args.cuda)

        if epoch % report_every == 0:
            logger.info('Guide Accuracy: {:.4f}'.format(
                    train_accuracy*100))

            val_accuracy = eval_epoch(X_valid, actions_valid, landmark_valid, y_valid,
                                      tourist, guide, args.batch_sz, args.cuda)
            test_accuracy = eval_epoch(X_test, actions_test, landmark_test, y_test,
                                       tourist, guide, args.batch_sz, args.cuda)

            val_acc.append(val_accuracy)
            test_acc.append(test_accuracy)

            logger.info('Valid Accuracy: {:.2f}% | Test Accuracy: {:.2f}%'.format(val_accuracy*100, test_accuracy*100))

            if val_accuracy > best_val_acc:
                tourist.save(os.path.join(exp_dir, 'tourist.pt'))
                guide.save(os.path.join(exp_dir, 'guide.pt'))
                best_val_acc = val_accuracy
                best_train_acc = train_accuracy
                best_test_acc = test_accuracy

    logger.info('%.2f, %.2f, %.2f' % (best_train_acc*100, best_val_acc*100, best_test_acc*100))

    plt.plot(range(len(train_acc)), train_acc, label='train')
    plt.plot(range(len(val_acc)), val_acc, label='valid')
    plt.plot(range(len(test_acc)), test_acc, label='test')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'curves.png'))