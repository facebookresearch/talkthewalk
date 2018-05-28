import argparse
import os
import random

import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from ttw.models import TouristLanguage
from ttw.data_loader import TalkTheWalkLanguage, get_collate_fn
from ttw.utils import create_logger
from ttw.dict import START_TOKEN, END_TOKEN

def show_samples(dataset, tourist, num_samples=10, cuda=True, logger=None, decoding_strategy='sample', indices=None):
    if indices is None:
        indices = list()
        for _ in range(num_samples):
            indices.append(random.randint(0, len(dataset)-1))

    collate_fn = get_collate_fn(cuda)

    data = [dataset[ind] for ind in indices]
    batch = collate_fn(data)

    out = tourist.forward(batch, decoding_strategy=decoding_strategy, train=False)

    generated_utterance = out['utterance'].cpu().data
    logger_fn = print
    if logger:
        logger_fn = logger

    for i in range(len(indices)):
        o = ''
        for obs in data[i]['observations']:
            o += '(' + ','.join([dataset.map.landmark_dict.decode(o_ind) for o_ind in obs]) + ') ,'
        # a = ', '.join([i2act[a_ind] for a_ind in actions[i]])
        a = ','.join([dataset.act_dict.decode(a_ind) for a_ind in data[i]['actions']])

        logger_fn('Observations: ' + o)
        logger_fn('Actions: ' + a)
        logger_fn('GT: ' + dataset.dict.decode(batch['utterance'][i, 1:]))
        logger_fn('Sample: ' + dataset.dict.decode(generated_utterance[i, :]))
        logger_fn('-'*80)

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
    return total_loss/total_examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--exp-dir', type=str, default='./exp')
    parser.add_argument('--act-emb-sz', type=int, default=128)
    parser.add_argument('--act-hid-sz', type=int, default=128)
    parser.add_argument('--obs-emb-sz', type=int, default=128)
    parser.add_argument('--obs-hid-sz', type=int, default=128)
    parser.add_argument('--decoder-emb-sz', type=int, default=128)
    parser.add_argument('--decoder-hid-sz', type=int, default=1024)
    parser.add_argument('--batch-sz', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--exp-name', type=str, default='tourist_sl')

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

    tourist = TouristLanguage(args.act_emb_sz, args.act_hid_sz, len(train_data.act_dict), args.obs_emb_sz, args.obs_hid_sz, len(train_data.map.landmark_dict),
                              args.decoder_emb_sz, args.decoder_hid_sz, len(train_data.dict),
                              start_token=train_data.dict.tok2i[START_TOKEN], end_token=train_data.dict.tok2i[END_TOKEN])

    opt = optim.Adam(tourist.parameters())

    if args.cuda:
        tourist = tourist.cuda()

    best_val = 1e10

    for epoch in range(1, args.num_epochs):
        train_loss = eval_epoch(train_loader, tourist, opt=opt)
        valid_loss = eval_epoch(valid_loader, tourist)

        logger.info('Epoch: {} \t Train loss: {},\t Valid_loss: {}'.format(epoch, train_loss, valid_loss))
        show_samples(valid_data, tourist, cuda=args.cuda, num_samples=5, logger=logger.info)

        if valid_loss < best_val:
            best_val = valid_loss
            tourist.save(os.path.join(exp_dir, 'tourist.pt'))






