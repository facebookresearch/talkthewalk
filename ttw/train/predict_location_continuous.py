import argparse
import os

import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from ttw.data_loader import TalkTheWalkEmergent, get_collate_fn
from ttw.models import TouristContinuous, GuideContinuous
from ttw.utils import create_logger

def epoch(loader, tourist, guide, opt=None):
    l, a = 0.0, 0.0
    n_batches = 0
    for batch in loader:
        msg = tourist.forward(batch)
        out = guide.forward(msg, batch)

        l += out['loss'].sum().item()
        a += out['acc']
        n_batches += 1

        if opt:
            opt.zero_grad()
            out['loss'].sum().backward()
            opt.step()
    return l/n_batches, a/n_batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--exp-dir', type=str, default='./exp')
    parser.add_argument('--apply-masc', action='store_true')
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--vocab-sz', type=int, default=500)
    parser.add_argument('--batch-sz', type=int, default=128)
    parser.add_argument('--report-every', type=int, default=5)
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--exp-name', type=str, default='test')
    parser.add_argument('--cuda', action='store_true')

    args = parser.parse_args()

    exp_dir = os.path.join(args.exp_dir, args.exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    args = parser.parse_args()

    logger = create_logger(os.path.join(exp_dir, 'log.txt'))
    logger.info(args)

    train_data = TalkTheWalkEmergent(args.data_dir, 'train', goldstandard_features=True, T=args.T)
    train_loader = DataLoader(train_data, args.batch_sz, collate_fn=get_collate_fn(args.cuda), shuffle=True)

    valid_data = TalkTheWalkEmergent(args.data_dir, 'valid', goldstandard_features=True, T=args.T)
    valid_loader = DataLoader(valid_data, args.batch_sz, collate_fn=get_collate_fn(args.cuda))

    test_data = TalkTheWalkEmergent(args.data_dir, 'test', goldstandard_features=True, T=args.T)
    test_loader = DataLoader(test_data, args.batch_sz, collate_fn=get_collate_fn(args.cuda))

    guide = GuideContinuous(args.vocab_sz, len(train_data.map.landmark_dict),
                            apply_masc=args.apply_masc, T=args.T)
    tourist = TouristContinuous(args.vocab_sz, len(train_data.map.landmark_dict),
                                apply_masc=args.apply_masc, T=args.T)

    params = list(tourist.parameters()) + list(guide.parameters())
    opt = optim.Adam(params)

    if args.cuda:
        tourist = tourist.cuda()
        guide = guide.cuda()

    best_train_loss, best_valid_loss, best_test_loss = None, 1e16, None
    best_train_acc, best_valid_acc, best_test_acc = None, None, None

    for i in range(1, args.num_epochs + 1):
        # train
        train_loss, train_acc = epoch(train_loader, tourist, guide, opt=opt)
        valid_loss, valid_acc = epoch(valid_loader, tourist, guide)
        test_loss, test_acc = epoch(test_loader, tourist, guide)

        logger.info("Train loss: {} | Valid loss: {} | Test loss: {}".format(train_loss,
                                                                       valid_loss,
                                                                       test_loss))
        logger.info("Train acc: {} | Valid acc: {} | Test acc: {}".format(train_acc,
                                                                    valid_acc,
                                                                    test_acc))

        if valid_loss < best_valid_loss:
            best_train_loss = train_loss
            best_valid_loss = valid_loss
            best_train_loss = test_loss

            best_train_acc, best_valid_acc, best_test_acc = train_acc, valid_acc, test_acc

            tourist.save(os.path.join(exp_dir, 'tourist.pt'))
            guide.save(os.path.join(exp_dir, 'guide.pt'))

    logger.info("%.2f, %.2f. %.2f" % (best_train_acc*100, best_valid_acc*100, best_test_acc*100))
