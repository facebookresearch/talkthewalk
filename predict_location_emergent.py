from __future__ import division

import os
import json
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.autograd import Variable
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from data_loader import Landmarks, create_batch, load_data, load_features, create_obs_dict, FasttextFeatures, GoldstandardFeatures, ResnetFeatures
from models import Tourist, Guide
from utils import create_logger

def eval_epoch(X, landmarks, y, tourist, guide, batch_sz):
    tourist.eval()
    guide.eval()

    correct, total = 0, 0
    for ii in range(0, len(y), args.batch_sz):
        X_batch = {k: X[k][ii:ii + batch_sz] for k in X.keys()}
        landmark_batch = landmarks[ii:ii + batch_sz]
        y_batch = y[ii:ii + batch_sz]
        batch_in, batch_landmarks, batch_mask, batch_tgt = create_batch(X_batch, landmark_batch, y_batch,
                                                                        cuda=args.cuda)

        # forward
        t_comms, t_probs, t_val = tourist(batch_in)
        t_msg = Variable(t_comms.data)
        if args.cuda:
            t_msg = t_msg.cuda()
        out_g = guide(t_msg, batch_landmarks, batch_mask)

        # acc
        correct += sum(
            [1.0 for pred, target in zip(torch.max(out_g, 1)[1].cpu().data, batch_tgt.cpu().data) if pred == target])
        total += len(y_batch)

    return correct/total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--resnet-features', action='store_true')
    parser.add_argument('--fasttext-features', action='store_true')
    parser.add_argument('--goldstandard-features', action='store_true')
    parser.add_argument('--softmax', choices=['landmarks', 'location'], default='location')
    parser.add_argument('--vocab-sz', type=int, default=10)
    parser.add_argument('--embed-sz', type=int, default=32)
    parser.add_argument('--batch-sz', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--exp-name', type=str, default='test')

    args = parser.parse_args()

    report_every = 10 # 10 epochs

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

    X_train, landmark_train, y_train = load_data(train_configs, feature_loaders, landmark_map, softmax=args.softmax)
    X_valid, landmark_valid, y_valid = load_data(valid_configs, feature_loaders, landmark_map, softmax=args.softmax)
    X_test, landmark_test, y_test = load_data(test_configs, feature_loaders, landmark_map, softmax=args.softmax)

    num_embeddings = len(landmark_map.types)+1
    if args.softmax == 'location':
        num_embeddings = len(landmark_map.global_coord_to_idx)
    # create models
    guide = Guide(args.vocab_sz, num_embeddings, embed_sz=args.embed_sz)
    tourist = Tourist(args.goldstandard_features, args.resnet_features, args.fasttext_features, args.vocab_sz, in_vocab_sz=in_vocab_sz, embed_sz=args.embed_sz)

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
        # X_train, landmark_train, y_train = shuffle(X_train, landmark_train, y_train)
        g_losses, t_rl_losses, t_val_losses = [], [], []
        g_accs = []
        for ii in range(0, len(y_train), args.batch_sz):
            X_batch = {k: X_train[k][ii:ii+args.batch_sz] for k in X_train.keys()}
            landmark_batch = landmark_train[ii:ii+args.batch_sz]
            y_batch = y_train[ii:ii+args.batch_sz]
            batch_in, batch_landmarks, batch_mask, batch_tgt = create_batch(X_batch, landmark_batch, y_batch, cuda=args.cuda)

            # forward
            t_comms, t_probs, t_val = tourist(batch_in)
            t_msg = Variable(t_comms.data)
            if args.cuda:
                t_msg = t_msg.cuda()
            out_g = guide(t_msg, batch_landmarks, batch_mask)

            # guide loss
            g_loss = -torch.log(torch.gather(out_g, 1, batch_tgt))
            _, max_ind = torch.max(out_g, 1)
            g_accs.extend([float(pred == target) for pred, target in zip(batch_tgt.cpu().data.numpy(), max_ind.cpu().data.numpy())])

            # tourist loss
            rewards = -g_loss # tourist reward is log likelihood of correct answer
            reward_std = np.maximum(np.std(rewards.cpu().data.numpy()), 1.)
            if reward_std == 0: reward_std = 1
            t_rl_loss, t_val_loss, t_reg = 0., 0., 0.
            eps, lamb = 1e-16, 1e-4
            for action, prob, val, reward in zip(t_comms, t_probs, t_val, rewards):
                advantage = Variable((reward.data - val.data) / reward_std)
                if args.cuda:
                    advantage = advantage.cuda()
                    action = action.cuda()
                action_prob = (action * prob) + ((1 - action) * (1 - prob))
                t_rl_loss -= torch.log(action_prob + eps)*advantage
                t_reg += lamb * (prob * action).norm(1) # increase sparsity? not sure if helps..
                t_val_loss += F.smooth_l1_loss(val, Variable(reward.data)) # mse: (val - Variable(reward.data)) ** 2
                # can add in supervised loss here to force grounding

            g_losses.append(g_loss.data.mean())
            t_rl_losses.append(t_rl_loss.data.mean())
            t_val_losses.append(t_val_loss.data.mean())

            # backward
            g_opt.zero_grad(); t_opt.zero_grad()
            g_loss.mean().backward()
            (t_rl_loss.mean() + t_val_loss.mean() + t_reg.mean()).backward()
            torch.nn.utils.clip_grad_norm(tourist.parameters(), 5)
            torch.nn.utils.clip_grad_norm(guide.parameters(), 5)
            g_opt.step(); t_opt.step()

        g_acc = sum(g_accs)/len(g_accs)

        if epoch % report_every == 0:
            train_acc.append(g_acc)
            logger.info('Guide Accuracy: {:.4f} | Guide loss: {:.4f} | Tourist loss: {:.4f} | Reward: {:.4f} | V: {:.4f}'.format( \
                    g_acc, np.mean(g_losses), np.mean(t_rl_losses), np.mean(rewards.cpu().data.numpy()), np.mean(t_val_losses)))

            val_accuracy = eval_epoch(X_valid, landmark_valid, y_valid, tourist, guide, args.batch_sz)
            test_accuracy = eval_epoch(X_test, landmark_test, y_test, tourist, guide, args.batch_sz)

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