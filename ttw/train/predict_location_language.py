# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os

import numpy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from ttw.data_loader import TalkTheWalkLanguage
from ttw.models import GuideLanguage
from ttw.logger import create_logger
from ttw.utils import get_collate_fn


def eval_epoch(loader, guide, opt=None):
    loss, accs, total = 0.0, 0.0, 0.0

    for batch in loader:
        g_out = guide.forward(batch, add_rl_loss=True)
        accs += g_out['acc']
        total += 1
        l = (g_out['rl_loss'] + g_out['sl_loss']).sum()
        loss += l.item()

        if opt is not None:
            opt.zero_grad()
            l.backward()
            opt.step()
    return loss/total, accs/total

def get_mean_T(loader, guide):
    distribution = numpy.array([0.0] * 4)
    for batch in loader:
        input_emb = guide.embed_fn(batch['utterance'])
        hidden_states, _ = guide.encoder_fn(input_emb)

        batch_sz = batch['utterance'].size(0)
        last_state_indices = batch['utterance_mask'].sum(1).long() - 1
        last_hidden_states = hidden_states[torch.arange(batch_sz).long(), last_state_indices, :]
        T_dist = F.softmax(guide.T_prediction_fn(last_hidden_states))

        distribution += T_dist.sum(0).cpu().data.numpy()
    distribution /= len(loader.dataset)
    mean_T = sum([p*v for p, v in zip(distribution, range(len(distribution)))])
    return mean_T


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to talkthewalk dataset')
    parser.add_argument('--exp-dir', type=str, default='./exp', help='Directory in which experiments will be stored')
    parser.add_argument('--exp-name', type=str, default='predict_location_nl',
                        help='Name of the experiment. Results will be stored in args.exp_dir/args.exp_name')
    parser.add_argument('--cuda', action='store_true', help='If true, runs on gpu')
    parser.add_argument('--apply-masc', action='store_true', help='If true, use MASC mechanism in the models')
    parser.add_argument('--T', type=int, default=2, help='Maximum *predicted* length of trajectory taken by the tourist')
    parser.add_argument('--hidden-sz', type=int, default=256, help='Number of hidden units of language encoder')
    parser.add_argument('--embed-sz', type=int, default=128, help='Word embedding size')
    parser.add_argument('--last-turns', type=int, default=1,
                        help='Specifies how many utterances from the dialogue are included to predict the location. '
                             'Note that guide utterances will be included as well.')
    parser.add_argument('--batch-sz', type=int, default=512, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs')

    args = parser.parse_args()

    exp_dir = os.path.join(args.exp_dir, args.exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    logger = create_logger(os.path.join(exp_dir, 'log.txt'))
    logger.info(args)

    train_data = TalkTheWalkLanguage(args.data_dir, 'train')
    train_loader = DataLoader(train_data, args.batch_sz, collate_fn=get_collate_fn(args.cuda), shuffle=True)

    valid_data = TalkTheWalkLanguage(args.data_dir, 'valid')
    valid_loader = DataLoader(valid_data, args.batch_sz, collate_fn=get_collate_fn(args.cuda))

    test_data = TalkTheWalkLanguage(args.data_dir, 'test')
    test_loader = DataLoader(test_data, args.batch_sz, collate_fn=get_collate_fn(args.cuda))


    guide = GuideLanguage(args.embed_sz, args.hidden_sz, len(train_data.dict), apply_masc=args.apply_masc, T=args.T)

    if args.cuda:
        guide = guide.cuda()
    opt = optim.Adam(guide.parameters())

    best_train_acc, best_val_acc, best_test_acc = 0.0, 0.0, 0.0
    for i in range(args.num_epochs):
        train_loss, train_acc = eval_epoch(train_loader, guide, opt=opt)
        valid_loss, valid_acc = eval_epoch(valid_loader, guide)
        test_loss, test_acc = eval_epoch(test_loader, guide)

        logger.info("Train loss: %.2f, Valid loss: %.2f, Test loss: %.2f" % (train_loss, valid_loss, test_loss))
        logger.info("Train acc: %.2f, Valid acc: %.2f, Test acc: %.2f" % (train_acc*100, valid_acc*100, test_acc*100))

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_train_acc, best_val_acc, best_test_acc = train_acc, valid_acc, test_acc
            guide.save(os.path.join(exp_dir, 'guide.pt'))

    logger.info(best_train_acc*100)
    logger.info(best_val_acc*100)
    logger.info(best_test_acc*100)

    best_guide = GuideLanguage.load(os.path.join(exp_dir, 'guide.pt'))
    if args.cuda:
        best_guide = best_guide.cuda()
    logger.info("mean T: {}".format(get_mean_T(test_loader, best_guide)))
