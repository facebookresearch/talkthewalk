from __future__ import division

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.distributions import Bernoulli
from torch.autograd import Variable
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from data_loader import Landmarks, load_data, load_features, create_obs_dict, FasttextFeatures, GoldstandardFeatures, ResnetFeatures
from utils import create_logger
from predict_location_continuous import create_batch
from modules import MASC, CBoW


def eval_epoch(X, actions, landmarks, y, tourist, guide, batch_sz):
    tourist.eval()
    guide.eval()

    correct, total = 0, 0
    for ii in range(0, len(y), args.batch_sz):
        X_batch = {k: X[k][ii:ii + args.batch_sz] for k in X.keys()}
        actions_batch = actions[ii:ii + args.batch_sz]
        landmark_batch = landmarks[ii:ii + args.batch_sz]
        y_batch = y[ii:ii + args.batch_sz]
        batch_in, batch_actions, batch_landmarks, batch_tgt = create_batch(X_batch, actions_batch, landmark_batch,
                                                                           y_batch, cuda=args.cuda)

        # forward
        t_comms, t_probs, t_val = tourist(batch_in, batch_actions)
        # t_msg = torch.cat(t_comms, 1).detach()
        # # t_msg = t_msg.type(torch.cuda.FloatTensor)
        # if args.cuda:
        #     t_msg = t_msg.cuda()
        out_g = guide(t_comms, batch_landmarks)

        # acc
        batch_tgt = batch_tgt[:, 0]*4 + batch_tgt[:, 1]
        correct += sum(
            [1.0 for pred, target in zip(torch.max(out_g, 1)[1].cpu().data, batch_tgt.cpu().data) if pred == target])
        total += len(y_batch)

    return correct/total


class Guide(nn.Module):

    def __init__(self, in_vocab_sz, num_landmarks, embed_sz=64, apply_masc=True, T=2):
        super(Guide, self).__init__()
        self.in_vocab_sz = in_vocab_sz
        self.num_landmarks = num_landmarks
        self.embed_sz = embed_sz
        self.T = T
        self.apply_masc = apply_masc
        self.emb_map = CBoW(num_landmarks, embed_sz, init_std=0.1)
        self.feature_emb = nn.ModuleList()
        for channel in range(T+1):
            self.feature_emb.append(nn.Linear(in_vocab_sz, embed_sz))


        self.masc_fn = MASC(embed_sz, apply_masc=apply_masc)
        if self.apply_masc:
            self.action_emb = nn.ModuleList()
            for i in range(T):
                self.action_emb.append(nn.Linear(in_vocab_sz, 9))

        self.act_lin = nn.Linear(embed_sz, 9)


    def forward(self, message, landmarks):
        f_msgs = list()
        for k in range(self.T+1):
            msg = message[0].cuda()
            # f_msgs.append(self.feature_emb.forward(message[:, k*self.num_channels:(k+1)*self.num_channels]))
            f_msgs.append(self.feature_emb[k].forward(msg))
        feature_msg = torch.cat(f_msgs, 1)
        batch_size = message[0].size(0)

        l_emb = self.emb_map.forward(landmarks).permute(0, 3, 1, 2)
        l_embs = [l_emb]

        if self.apply_masc:
            for j in range(self.T):
                action_out = self.action_emb[j](message[1].cuda())

                out = self.masc_fn.forward(l_embs[-1], action_out)
                l_embs.append(out)
        else:
            for j in range(self.T):
                out = self.masc_fn.forward_no_masc(l_embs[-1])
                l_embs.append(out)

        landmarks = torch.cat(l_embs, 1)
        landmarks = landmarks.view(batch_size, landmarks.size(1), 16).transpose(1, 2)

        logits = torch.bmm(landmarks, feature_msg.unsqueeze(-1)).squeeze(-1)
        prob = F.softmax(logits, dim=1)
        return prob


    def save(self, path):
        state = dict()
        state['in_vocab_sz'] = self.in_vocab_sz
        state['num_landmarks'] = self.num_landmarks
        state['embed_sz'] = self.embed_sz
        state['parameters'] = self.state_dict()
        state['T'] = self.T
        state['apply_masc'] = self.apply_masc
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        guide = cls(state['in_vocab_sz'], state['num_landmarks'], embed_sz=state['embed_sz'], T=state['T'],
                    apply_masc=state['apply_masc'])
        guide.load_state_dict(state['parameters'])
        return guide


class Tourist(nn.Module):
    def __init__(self, goldstandard_features, resnet_features, fasttext_features,
                 emb_sz, vocab_sz, T=2, apply_masc=False):
        super(Tourist, self).__init__()
        self.goldstandard_features = goldstandard_features
        self.resnet_features = resnet_features
        self.fasttext_features = fasttext_features
        self.emb_sz = emb_sz
        self.T = T
        self.apply_masc = apply_masc
        self.vocab_sz = vocab_sz


        if self.goldstandard_features:
            self.goldstandard_emb = nn.Embedding(11, emb_sz)
        if self.fasttext_features:
            self.fasttext_emb_linear = nn.Linear(300, emb_sz)
        if self.resnet_features:
            self.resnet_emb_linear = nn.Linear(2048, emb_sz)

        self.num_embeddings = T+1
        if self.apply_masc:
            self.action_emb = nn.Embedding(4, emb_sz)
            self.num_embeddings += T
            self.act_comms = nn.Linear(T * self.emb_sz, vocab_sz)

        self.feat_comms = nn.Linear((T+1)*self.emb_sz, vocab_sz)

        self.loss = nn.CrossEntropyLoss()
        self.value_pred = nn.Linear(self.num_embeddings*self.emb_sz, 1)

    def forward(self, X, actions, greedy=False):
        batch_size = actions.size(0)
        feat_emb = list()
        if self.goldstandard_features:
            max_steps = X['goldstandard'].size(1)
            for step in range(max_steps):
                feat_emb.append(self.goldstandard_emb.forward(X['goldstandard'][:, step, :]).sum(dim=1))

        act_emb = list()
        if self.apply_masc:
            for step in range(self.T):
                act_emb.append(self.action_emb.forward(actions[:, step]))

        comms = list()
        probs = list()

        feat_embeddings = torch.cat(feat_emb, 1)
        feat_logits = self.feat_comms(feat_embeddings)
        # feat_logits = feat_embeddings
        feat_prob = F.sigmoid(feat_logits).cpu()
        feat_msg = feat_prob.bernoulli().detach()

        probs.append(feat_prob)
        comms.append(feat_msg)

        if self.apply_masc:
            act_embeddings = torch.cat(act_emb, 1)
            # act_logits = act_embeddings
            act_logits = self.act_comms(act_embeddings)
            act_prob = F.sigmoid(act_logits).cpu()
            act_msg = act_prob.bernoulli().detach()

            probs.append(act_prob)
            comms.append(act_msg)

        # emb = torch.cat(embeddings, 1)
        if self.apply_masc:
            embeddings = torch.cat([feat_embeddings, act_embeddings], 1).resize(batch_size, self.num_embeddings*self.emb_sz)
        else:
            embeddings = torch.cat([feat_embeddings], 1).resize(batch_size, self.num_embeddings * self.emb_sz)
        value = self.value_pred(embeddings)

        return comms, probs, value

    def save(self, path):
        state = dict()
        state['goldstandard_features'] = self.goldstandard_features
        state['resnet_features'] = self.resnet_features
        state['fasttext_features'] = self.fasttext_features
        state['emb_sz'] = self.emb_sz
        state['T'] = self.T
        state['apply_masc'] = self.apply_masc
        state['parameters'] = self.state_dict()
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        tourist = cls(state['goldstandard_features'], state['resnet_features'], state['fasttext_features'],
                      state['emb_sz'], 1000, T=state['T'], apply_masc=state['condition_on_action'])
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
    parser.add_argument('--vocab-sz', type=int, default=2)
    parser.add_argument('--embed-sz', type=int, default=32)
    parser.add_argument('--batch-sz', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--exp-name', type=str, default='test')

    args = parser.parse_args()

    report_every = 5 # 10 epochs

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
        in_vocab_sz = len(landmark_map.itos) + 1
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

    num_embeddings = len(landmark_map.types)+1
    if args.softmax == 'location':
        num_embeddings = len(landmark_map.global_coord_to_idx)
    # create models
    in_vocab_sz = (args.T+1)*args.embed_sz
    if args.masc:
        in_vocab_sz += args.T*args.embed_sz

    guide = Guide(args.vocab_sz, num_embeddings, embed_sz=args.embed_sz,
                  apply_masc=args.masc, T=args.T)
    tourist = Tourist(args.goldstandard_features, args.resnet_features, args.fasttext_features, args.embed_sz, args.vocab_sz,
                      apply_masc=args.masc, T=args.T)

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
        tourist.train(); guide.train()
        X_train['goldstandard'], actions_train, landmark_train, y_train = shuffle(X_train['goldstandard'], actions_train, landmark_train, y_train)
        g_losses, t_rl_losses, t_val_losses = [], [], []
        g_accs = []
        for ii in range(0, len(y_train), args.batch_sz):
            X_batch = {k: X_train[k][ii:ii+args.batch_sz] for k in X_train.keys()}
            actions_batch = actions_train[ii:ii+args.batch_sz]
            landmark_batch = landmark_train[ii:ii+args.batch_sz]
            y_batch = y_train[ii:ii+args.batch_sz]
            batch_in, batch_actions, batch_landmarks, batch_tgt = create_batch(X_batch, actions_batch, landmark_batch, y_batch, cuda=args.cuda)

            # forward
            t_comms, t_probs, t_val = tourist(batch_in, batch_actions)
            # t_msg = torch.cat(t_comms, 1).detach()
            # t_msg = t_msg.type(torch.cuda.FloatTensor)
            # if args.cuda:
            #     t_msg = t_msg.cuda()
            out_g = guide(t_comms, batch_landmarks)

            # guide loss
            batch_tgt = batch_tgt[:, 0]*4 + batch_tgt[:, 1]
            g_loss = -torch.log(torch.gather(out_g, 1, batch_tgt))
            _, max_ind = torch.max(out_g, 1)
            g_accs.extend([float(pred == target) for pred, target in zip(batch_tgt.cpu().data.numpy(), max_ind.cpu().data.numpy())])

            # tourist loss
            rewards = -g_loss # tourist reward is log likelihood of correct answer
            reward_std = np.maximum(np.std(rewards.cpu().data.numpy()), 1.)
            reward_mean = float(rewards.cpu().data.numpy().mean())
            if reward_std == 0: reward_std = 1
            t_rl_loss, t_val_loss, t_reg = 0., 0., 0.
            eps, lamb = 1e-16, 1e-4

            advantage = Variable((rewards.data - t_val.data))
            if args.cuda:
                advantage = advantage.cuda()
            t_val_loss += ((t_val - Variable(rewards.data)) ** 2).mean()  # mse

            for action, prob in zip(t_probs, t_probs):
                if args.cuda:
                    action = action.cuda()
                    prob = prob.cuda()
                action_prob = action*prob + (1.0-action)*(1.0-prob)

                # action_prob = torch.gather(prob, 1, action)
                t_rl_loss -= (torch.log(action_prob + eps)*advantage).sum(1)
                # t_reg += lamb * (torch.log(prob)*prob)
                # t_reg += lamb * (prob * action).norm(1) # increase sparsity? not sure if helps..
                # t_val_loss += F.smooth_l1_loss(val, Variable(reward.data))

                # can add in supervised loss here to force grounding

            g_losses.append(g_loss.data.mean())
            t_rl_losses.append(t_rl_loss.data.mean())
            t_val_losses.append(t_val_loss.data.mean())

            # backward
            g_opt.zero_grad(); t_opt.zero_grad()
            g_loss.mean().backward()
            (t_rl_loss.mean() + t_val_loss.mean()).backward()
            torch.nn.utils.clip_grad_norm(tourist.parameters(), 5)
            torch.nn.utils.clip_grad_norm(guide.parameters(), 5)
            g_opt.step(); t_opt.step()

        g_acc = sum(g_accs)/len(g_accs)

        if epoch % report_every == 0:
            train_acc.append(g_acc)
            logger.info('Guide Accuracy: {:.4f} | Guide loss: {:.4f} | Tourist loss: {:.4f} | Reward: {:.4f} | V: {:.4f}'.format( \
                    g_acc, np.mean(g_losses), np.mean(t_rl_losses), np.mean(rewards.cpu().data.numpy()), np.mean(t_val_losses)))

            val_accuracy = eval_epoch(X_valid, actions_valid, landmark_valid, y_valid, tourist, guide, args.batch_sz)
            test_accuracy = eval_epoch(X_test, actions_test, landmark_test, y_test, tourist, guide, args.batch_sz)

            val_acc.append(val_accuracy)
            test_acc.append(test_accuracy)

            logger.info('Valid Accuracy: {:.2f}% | Test Accuracy: {:.2f}%'.format(val_accuracy*100, test_accuracy*100))

            if val_accuracy > best_val_acc:
                tourist.save(os.path.join(exp_dir, 'tourist.pt'))
                guide.save(os.path.join(exp_dir, 'guide.pt'))
                best_val_acc = val_accuracy
                best_train_acc = g_acc
                best_test_acc = test_accuracy

    print(best_train_acc, best_val_acc, best_test_acc)

    plt.plot(range(len(train_acc)), train_acc, label='train')
    plt.plot(range(len(val_acc)), val_acc, label='valid')
    plt.plot(range(len(test_acc)), test_acc, label='test')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'curves.png'))