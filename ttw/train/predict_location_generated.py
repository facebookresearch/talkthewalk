import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from ttw.data_loader import TalkTheWalkLanguage, get_collate_fn
from ttw.models import GuideLanguage, TouristLanguage
from ttw.utils import create_logger

def cache():
    raise NotImplementedError()

def epoch(loader, tourist, guide, g_opt=None, t_opt=None, cached=True,
          decoding_strategy='beam_search'):
    accuracy, total = 0.0, 0.0

    for batch in loader:

        out = tourist.forward(batch,
                              decoding_strategy=decoding_strategy, train=False)
        batch['utterance'] = out['preds']
        batch['utterance_mask'] = out['mask']

        loss, acc = guide.forward(batch)

        reward = -loss.squeeze()
        loss = loss.sum()

        total += batch['landmarks'].size(0)
        accuracy += acc * batch['landmarks'].size(0)

        if g_opt is not None:
            g_opt.zero_grad()
            loss.backward()
            g_opt.step()

        if t_opt is not None:
            # reinforce
            probs = out['probs']
            mask = out['mask']
            sampled_ind = out['preds']

            loss = 0.0

            for k in range(probs.size(1)):
                prob = probs[:, k, :]
                ind = sampled_ind[:, k]
                selected_prob = torch.gather(prob, 1, ind.unsqueeze(-1))
                log_prob = torch.log(selected_prob + 1e-8).squeeze(-1)
                advantage = reward - reward.mean()
                loss -= (mask[:, k] * log_prob * advantage).sum()

            t_opt.zero_grad()
            loss.backward()
            t_opt.step()

    return accuracy / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--exp-dir', type=str, default='./exp')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--on-the-fly', action='store_true',
                        help="Generate samples from tourist model on the fly. If not, samples are cached")
    parser.add_argument('--tourist-model', type=str)
    parser.add_argument('--guide-model', type=str)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-sz', type=int, default=512)
    parser.add_argument('--train-guide', action='store_true')
    parser.add_argument('--train-tourist', action='store_true')
    parser.add_argument('--exp-name', default='test')
    parser.add_argument('--decoding-strategy', type=str)

    args = parser.parse_args()

    exp_dir = os.path.join(args.exp_dir, args.exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    logger = create_logger(os.path.join(exp_dir, 'log.txt'))
    logger.info(args)

    data_dir = args.data_dir

    train_data = TalkTheWalkLanguage(data_dir, 'train')
    train_loader = DataLoader(train_data, args.batch_sz, collate_fn=get_collate_fn(args.cuda))

    valid_data = TalkTheWalkLanguage(data_dir, 'valid')
    valid_loader = DataLoader(valid_data, args.batch_sz, collate_fn=get_collate_fn(args.cuda))

    test_data = TalkTheWalkLanguage(data_dir, 'test')
    test_loader = DataLoader(test_data, args.batch_sz, collate_fn=get_collate_fn(args.cuda))

    tourist = TouristLanguage.load(args.tourist_model)
    if args.guide_model is not None:
        guide = GuideLanguage.load(args.guide_model)
    else:
        guide = GuideLanguage(128, 256, len(train_data.dict), apply_masc=True, T=3)

    if args.cuda:
        tourist = tourist.cuda()
        guide = guide.cuda()

    if args.train_guide:
        logger.info('Train guide (supervised)')
        g_opt = optim.Adam(guide.parameters())

    if args.train_tourist:
        logger.info('Train tourist (supervised)')
        t_opt = optim.Adam(tourist.parameters())


    if not args.on_the_fly:
        raise NotImplementedError()

    best_train_acc, best_valid_acc, best_test_acc = 0.0, 0.0, 0.0
    for i in range(args.num_epochs):
        g_optim = None
        if args.train_guide:
            g_optim = g_opt

        t_optim = None
        if args.train_tourist:
            t_optim = t_opt

        train_acc = epoch(train_loader, tourist, guide, g_opt=g_optim, t_opt=t_optim,
                          decoding_strategy=args.decoding_strategy)
        valid_acc = epoch(valid_loader, tourist, guide, decoding_strategy=args.decoding_strategy)
        test_acc = epoch(test_loader, tourist, guide, decoding_strategy=args.decoding_strategy)

        logger.info(
            'Epoch: {} -- Train acc: {}, Valid acc: {}, Test acc: {}'.format(i + 1, train_acc * 100, valid_acc * 100,
                                                                             test_acc * 100))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            tourist.save(os.path.join(exp_dir, 'tourist.pt'))
            guide.save(os.path.join(exp_dir, 'guide.pt'))
