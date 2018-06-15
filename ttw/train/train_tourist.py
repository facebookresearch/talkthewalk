# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import random

import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from ttw.models import TouristLanguage
from ttw.data_loader import TalkTheWalkLanguage
from ttw.logger import create_logger
from ttw.dict import START_TOKEN, END_TOKEN
from ttw.utils import get_collate_fn


def eval_epoch(loader, tourist, opt=None):
    total_loss, total_examples = 0.0, 0.0
    for batch in loader:
        out = tourist.forward(batch,
                              train=True)
        loss = out['loss']
        total_loss += float(loss.data)
        total_examples += batch['utterance'].size(0)

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
    return total_loss / total_examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to talkthewalk dataset')
    parser.add_argument('--exp-dir', type=str, default='./exp', help='Directory in which experiments will be stored')
    parser.add_argument('--exp-name', type=str, default='tourist_sl',
                        help='Name of the experiment. Results will be stored in args.exp_dir/args.exp_name')
    parser.add_argument('--cuda', action='store_true', help='If true, runs on gpu')
    parser.add_argument('--act-emb-sz', type=int, default=128, help='Dimensionality of action embedding')
    parser.add_argument('--act-hid-sz', type=int, default=128, help='Dimensionality of action encoder')
    parser.add_argument('--obs-emb-sz', type=int, default=128, help='Dimensionality of observation embedding')
    parser.add_argument('--obs-hid-sz', type=int, default=128, help='Dimensionality of observation encoder')
    parser.add_argument('--decoder-emb-sz', type=int, default=128, help='Dimensionality of word embeddings')
    parser.add_argument('--decoder-hid-sz', type=int, default=1024, help='Hidden size of decoder RNN')
    parser.add_argument('--batch-sz', type=int, default=128, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')

    args = parser.parse_args()

    exp_dir = os.path.join(args.exp_dir, args.exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    logger = create_logger(os.path.join(exp_dir, 'log.txt'))
    logger.info(args)

    data_dir = args.data_dir

    train_data = TalkTheWalkLanguage(data_dir, 'train')
    train_loader = DataLoader(train_data, args.batch_sz, shuffle=True, collate_fn=get_collate_fn(args.cuda))

    valid_data = TalkTheWalkLanguage(data_dir, 'valid')
    valid_loader = DataLoader(valid_data, args.batch_sz, collate_fn=get_collate_fn(args.cuda))

    tourist = TouristLanguage(args.act_emb_sz, args.act_hid_sz, len(train_data.act_dict), args.obs_emb_sz,
                              args.obs_hid_sz, len(train_data.map.landmark_dict),
                              args.decoder_emb_sz, args.decoder_hid_sz, len(train_data.dict),
                              start_token=train_data.dict.tok2i[START_TOKEN],
                              end_token=train_data.dict.tok2i[END_TOKEN])

    opt = optim.Adam(tourist.parameters())

    if args.cuda:
        tourist = tourist.cuda()

    best_val = 1e10

    for epoch in range(1, args.num_epochs):
        train_loss = eval_epoch(train_loader, tourist, opt=opt)
        valid_loss = eval_epoch(valid_loader, tourist)

        logger.info('Epoch: {} \t Train loss: {},\t Valid_loss: {}'.format(epoch, train_loss, valid_loss))
        tourist.show_samples(valid_data, cuda=args.cuda, num_samples=5, logger=logger.info)

        if valid_loss < best_val:
            best_val = valid_loss
            tourist.save(os.path.join(exp_dir, 'tourist.pt'))
