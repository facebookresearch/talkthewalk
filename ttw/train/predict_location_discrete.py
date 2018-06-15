# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from ttw.data_loader import TalkTheWalkEmergent
from ttw.models import TouristDiscrete, GuideDiscrete
from ttw.logger import create_logger
from ttw.utils import get_collate_fn

def eval_epoch(loader, tourist, guide, cuda, t_opt=None, g_opt=None):
    tourist.eval()
    guide.eval()

    correct, total = 0, 0
    for batch in loader:
        # forward
        t_out = tourist(batch)
        if cuda:
            t_out['comms'] = [x.cuda() for x in t_out['comms']]
        g_out = guide(t_out['comms'], batch)

        # acc
        correct += g_out['acc']*len(batch['target'])
        total += len(batch['target'])

        if t_opt and g_opt:
            # train if optimizers are specified
            rewards = -g_out['loss'].unsqueeze(-1)  # tourist reward is log likelihood of correct answer

            t_rl_loss = 0.
            eps = 1e-16

            advantage = Variable((rewards.data - t_out['baseline'].data))
            if cuda:
                advantage = advantage.cuda()
            t_val_loss = ((t_out['baseline'] - Variable(rewards.data)) ** 2).mean()  # mse

            for action, prob in zip(t_out['comms'], t_out['probs']):
                if cuda:
                    action = action.cuda()
                    prob = prob.cuda()

                action_prob = action * prob + (1.0 - action) * (1.0 - prob)

                t_rl_loss -= (torch.log(action_prob + eps) * advantage).sum()

            # backward
            g_opt.zero_grad()
            t_opt.zero_grad()
            g_out['loss'].sum().backward()
            (t_rl_loss + t_val_loss).backward()
            torch.nn.utils.clip_grad_norm(tourist.parameters(), 5)
            torch.nn.utils.clip_grad_norm(guide.parameters(), 5)
            g_opt.step()
            t_opt.step()

    return correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to talkthewalk dataset')
    parser.add_argument('--exp-dir', type=str, default='./exp', help='Directory in which experiments will be stored')
    parser.add_argument('--exp-name', type=str, default='predict_location_discrete',
                        help='Name of the experiment. Results will be stored in args.exp_dir/args.exp_name')
    parser.add_argument('--cuda', action='store_true', help='If true, runs on gpu')
    parser.add_argument('--apply-masc', action='store_true', help='If true, use MASC mechanism in the models')
    parser.add_argument('--T', type=int, default=2, help='Length of trajectory taken by the tourist')
    parser.add_argument('--vocab-sz', type=int, default=500,
                        help='Dimension of the observation and action embedding send from tourist to guide')
    parser.add_argument('--batch-sz', type=int, default=128)
    parser.add_argument('--report-every', type=int, default=5)
    parser.add_argument('--num-epochs', type=int, default=400, help='Number of epochs')


    args = parser.parse_args()

    exp_dir = os.path.join(args.exp_dir, args.exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    logger = create_logger(os.path.join(exp_dir, 'log.txt'))
    logger.info(args)

    train_data = TalkTheWalkEmergent(args.data_dir, 'train', goldstandard_features=True, T=args.T)
    train_loader = DataLoader(train_data, args.batch_sz, collate_fn=get_collate_fn(args.cuda), shuffle=True)

    valid_data = TalkTheWalkEmergent(args.data_dir, 'valid', goldstandard_features=True, T=args.T)
    valid_loader = DataLoader(valid_data, args.batch_sz, collate_fn=get_collate_fn(args.cuda))

    test_data = TalkTheWalkEmergent(args.data_dir, 'test', goldstandard_features=True, T=args.T)
    test_loader = DataLoader(test_data, args.batch_sz, collate_fn=get_collate_fn(args.cuda))

    guide = GuideDiscrete(args.vocab_sz, len(train_data.map.landmark_dict),
                          apply_masc=args.apply_masc, T=args.T)
    tourist = TouristDiscrete(args.vocab_sz, len(train_data.map.landmark_dict), len(train_data.act_dict),
                              apply_masc=args.apply_masc, T=args.T)

    if args.cuda:
        guide = guide.cuda()
        tourist = tourist.cuda()

    g_opt, t_opt = optim.Adam(guide.parameters()), optim.Adam(tourist.parameters())

    train_acc = list()
    val_acc = list()
    test_acc = list()

    best_train_acc, best_val_acc, best_test_acc = 0.0, 0.0, 0.0

    for epoch in range(1, args.num_epochs):
        train_accuracy = eval_epoch(train_loader, tourist, guide, args.cuda,
                                    t_opt=t_opt, g_opt=g_opt)

        if epoch % args.report_every == 0:
            logger.info('Guide Accuracy: {:.4f}'.format(
                train_accuracy * 100))

            val_accuracy = eval_epoch(valid_loader, tourist, guide, args.cuda)
            test_accuracy = eval_epoch(test_loader, tourist, guide, args.cuda)

            val_acc.append(val_accuracy)
            test_acc.append(test_accuracy)

            logger.info(
                'Valid Accuracy: {:.2f}% | Test Accuracy: {:.2f}%'.format(val_accuracy * 100, test_accuracy * 100))

            if val_accuracy > best_val_acc:
                tourist.save(os.path.join(exp_dir, 'tourist.pt'))
                guide.save(os.path.join(exp_dir, 'guide.pt'))
                best_val_acc = val_accuracy
                best_train_acc = train_accuracy
                best_test_acc = test_accuracy

    logger.info('%.2f, %.2f, %.2f' % (best_train_acc * 100, best_val_acc * 100, best_test_acc * 100))
