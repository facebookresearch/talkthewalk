from __future__ import division

import os
import math
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

from data_loader import Landmarks, load_data_multiple_step, load_features, create_obs_dict, FasttextFeatures, GoldstandardFeatures, ResnetFeatures
from utils import create_logger
from predict_location_continuous import MapEmbedding2d, create_batch

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

    def __init__(self, in_vocab_sz, num_landmarks, embed_sz=64, condition_on_action=True, max_steps=2):
        super(Guide, self).__init__()
        self.in_vocab_sz = in_vocab_sz
        self.num_landmarks = num_landmarks
        self.embed_sz = embed_sz
        self.max_steps = max_steps
        self.condition_on_action = condition_on_action
        self.emb_map = MapEmbedding2d(num_landmarks, embed_sz, init_std=0.1)
        self.feature_emb = nn.ModuleList()
        for channel in range(max_steps):
            self.feature_emb.append(nn.Linear(in_vocab_sz, embed_sz))

        self.conv_weight = nn.Parameter(torch.FloatTensor(
                embed_sz, embed_sz, 3, 3))
        std = 1.0/(embed_sz*9)
        self.conv_weight.data.uniform_(-std, std)
        self.mask_conv = True
        if self.condition_on_action:
            self.action_emb = nn.ModuleList()
            for i in range(max_steps - 1):
                self.action_emb.append(nn.Linear(in_vocab_sz, 9))

        self.attention = nn.Parameter(torch.FloatTensor(2*max_steps-1, embed_sz).normal_(0.0, 0.1))
        self.act_lin = nn.Linear(embed_sz, 9)


    def forward(self, message, landmarks):
        # batch_size = message.size(0)
        # feat_embs = list()
        # for c in range(self.num_channels):
        #     feat_embs.append(self.feature_emb[c](message[:, c]).unsqueeze(1))
        # feat_emb = torch.cat(feat_embs, 1)
        # feature_msg = feat_emb.sum(1)

        # f_msgs = list()
        # for k in range(self.max_steps):
        #     query = self.attention[k, :].unsqueeze(-1).unsqueeze(0).repeat(batch_size, 1, 1)
        #     score = torch.bmm(feat_emb, query).squeeze(-1)
        #     att = F.softmax(score, dim=1).unsqueeze(1)
        #     f_emb = torch.bmm(att, feat_emb).squeeze(1)
        #     f_msgs.append(f_emb)
        # feature_msg = torch.cat(f_msgs, 1)

        f_msgs = list()
        for k in range(self.max_steps):
            msg = message[0].cuda()
            # f_msgs.append(self.feature_emb.forward(message[:, k*self.num_channels:(k+1)*self.num_channels]))
            f_msgs.append(self.feature_emb[k].forward(msg))
        feature_msg = torch.cat(f_msgs, 1)
        batch_size = message[0].size(0)

        l_emb = self.emb_map.forward(landmarks).permute(0, 3, 1, 2)
        l_embs = [l_emb]

        if self.condition_on_action:
            for j in range(self.max_steps - 1):
                k = self.max_steps + j
                # query = self.attention[k, :].unsqueeze(-1).unsqueeze(0).repeat(batch_size, 1, 1)
                # score = torch.bmm(feat_emb, query).squeeze(-1)
                # att = F.softmax(score, dim=1).unsqueeze(1)
                # action_out = torch.bmm(att, feat_emb).squeeze(1)
                action_out = self.action_emb[j](message[1].cuda())
                # action_out = self.act_lin.forward(action_out)

                # act_emb = list()
                # for channel in range(self.num_channels):
                #     act_emb.append(self.action_emb[channel].forward(message[:, k*self.num_channels+channel]))
                # action_out = sum(act_emb)
                # action_out = self.action_emb.forward(message[:, k*self.num_channels:(k+1)*self.num_channels])
                out = Variable(torch.FloatTensor(batch_size, self.embed_sz, 4, 4).zero_().cuda())
                for i in range(batch_size):
                    selected_inp = l_embs[-1][i, :, :, :].unsqueeze(0)
                    mask = F.softmax(action_out[i], dim=0).resize(1, 1, 3, 3)
                    weight = mask * self.conv_weight
                    out[i, :, :, :] = F.conv2d(selected_inp, weight, padding=1).squeeze(0)
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
            for j in range(self.max_steps - 1):
                tmp = F.conv2d(l_embs[-1], weight, padding=1)
                l_embs.append(tmp)

        landmarks = torch.cat(l_embs, 1)
        landmarks = landmarks.resize(batch_size, landmarks.size(1), 16).transpose(1, 2)


        logits = torch.bmm(landmarks, feature_msg.unsqueeze(-1)).squeeze(-1)
        prob = F.softmax(logits, dim=1)
        return prob


    def save(self, path):
        state = dict()
        state['in_vocab_sz'] = self.in_vocab_sz
        state['num_landmarks'] = self.num_landmarks
        state['embed_sz'] = self.embed_sz
        state['parameters'] = self.state_dict()
        state['max_steps'] = self.max_steps
        state['condition_on_action'] = self.condition_on_action
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        guide = cls(state['in_vocab_sz'], state['num_landmarks'], embed_sz=state['embed_sz'], max_steps=state['max_steps'],
                    condition_on_action=state['condition_on_action'])
        guide.load_state_dict(state['parameters'])
        return guide


class Tourist(nn.Module):
    def __init__(self, goldstandard_features, resnet_features, fasttext_features,
                 emb_sz, vocab_sz, max_steps=2, condition_on_action=False):
        super(Tourist, self).__init__()
        self.goldstandard_features = goldstandard_features
        self.resnet_features = resnet_features
        self.fasttext_features = fasttext_features
        self.emb_sz = emb_sz
        self.max_steps = max_steps
        self.condition_on_action = condition_on_action
        self.vocab_sz = vocab_sz


        if self.goldstandard_features:
            self.goldstandard_emb = nn.Embedding(11, emb_sz)
        if self.fasttext_features:
            self.fasttext_emb_linear = nn.Linear(300, emb_sz)
        if self.resnet_features:
            self.resnet_emb_linear = nn.Linear(2048, emb_sz)

        self.num_embeddings = max_steps
        if self.condition_on_action:
            self.action_emb = nn.Embedding(4, emb_sz)
            self.num_embeddings += max_steps - 1
            self.act_comms = nn.Linear((max_steps - 1) * self.emb_sz, vocab_sz)

        self.feat_comms = nn.Linear(self.max_steps*self.emb_sz, vocab_sz)

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
        if self.condition_on_action:
            for step in range(max_steps-1):
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

        if self.condition_on_action:
            act_embeddings = torch.cat(act_emb, 1)
            # act_logits = act_embeddings
            act_logits = self.act_comms(act_embeddings)
            act_prob = F.sigmoid(act_logits).cpu()
            act_msg = act_prob.bernoulli().detach()

            probs.append(act_prob)
            comms.append(act_msg)

        # probs = list()
        # comms = list()
        # sampled_actions = list()


        # embeddings = emb.resize(batch_size, self.num_embeddings*self.emb_sz)
        # # embeddings = list()
        # for k in range(self.length):
        #     # query = self.attention_score[k, :].unsqueeze(-1).unsqueeze(0).repeat(batch_size, 1, 1)
        #     # score = torch.bmm(emb, query).squeeze(-1)
        #     # att = F.softmax(score, dim=1)
        #     #
        #     # # # soft attention
        #     # e = torch.bmm(att.unsqueeze(1), emb).squeeze(1)
        #     # embeddings.append(e)
        #
        #     # hard attention
        #     # att = att.squeeze()
        #     # sampled_index = att.cpu().multinomial(1)
        #     # sampled_actions.append(sampled_index)
        #     # probs.append(att)
        #     # e = torch.cat([emb[i, sampled_index[i, 0].cuda(), :] for i in range(batch_size)], 0)
        #     # embeddings.append(e)
        #
        #     # e = emb[:, k, :]
        #     # embeddings.append(e)
        #
        #     logits = self.out_comms[k](emb[:, k, :])
        #     prob = F.softmax(logits, dim=-1)
        #     probs.append(prob)
        #     sampled_comm = prob.cpu().multinomial(1)
        #     comms.append(sampled_comm)
        #     sampled_actions.append(sampled_comm)

        # emb = torch.cat(embeddings, 1)
        if self.condition_on_action:
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
        state['max_steps'] = self.max_steps
        state['condition_on_action'] = self.condition_on_action
        state['parameters'] = self.state_dict()
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        tourist = cls(state['goldstandard_features'], state['resnet_features'], state['fasttext_features'],
                      state['emb_sz'], 1000, max_steps=state['max_steps'], condition_on_action=state['condition_on_action'])
        tourist.load_state_dict(state['parameters'])

        return tourist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--resnet-features', action='store_true')
    parser.add_argument('--fasttext-features', action='store_true')
    parser.add_argument('--goldstandard-features', action='store_true')
    parser.add_argument('--softmax', choices=['landmarks', 'location'], default='landmarks')
    parser.add_argument('--condition-on-action', action='store_true')
    parser.add_argument('--num-steps', type=int, default=2)
    parser.add_argument('--vocab-sz', type=int, default=2)
    parser.add_argument('--embed-sz', type=int, default=32)
    parser.add_argument('--batch-sz', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--exp-name', type=str, default='test')

    args = parser.parse_args()

    report_every = 5 # 10 epochs

    exp_dir = os.path.join(os.environ['TALKTHEWALK_EXPDIR'], args.exp_name)
    if os.path.exists(exp_dir):
        raise RuntimeError('Experiment directory already exist..')
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

    X_train, actions_train, landmark_train, y_train = load_data_multiple_step(train_configs, feature_loaders,
                                                                              landmark_map,
                                                                              softmax=args.softmax,
                                                                              num_steps=args.num_steps)
    X_valid, actions_valid, landmark_valid, y_valid = load_data_multiple_step(valid_configs, feature_loaders,
                                                                              landmark_map,
                                                                              softmax=args.softmax,
                                                                              num_steps=args.num_steps)
    X_test, actions_test, landmark_test, y_test = load_data_multiple_step(test_configs, feature_loaders, landmark_map,
                                                                          softmax=args.softmax,
                                                                          num_steps=args.num_steps)

    num_embeddings = len(landmark_map.types)+1
    if args.softmax == 'location':
        num_embeddings = len(landmark_map.global_coord_to_idx)
    # create models
    in_vocab_sz = args.num_steps*args.embed_sz
    if args.condition_on_action:
        in_vocab_sz += (args.num_steps - 1)*args.embed_sz

    guide = Guide(args.vocab_sz, num_embeddings, embed_sz=args.embed_sz,
                  condition_on_action=args.condition_on_action, max_steps=args.num_steps)
    tourist = Tourist(args.goldstandard_features, args.resnet_features, args.fasttext_features, args.embed_sz, args.vocab_sz,
                      condition_on_action=args.condition_on_action, max_steps=args.num_steps)

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