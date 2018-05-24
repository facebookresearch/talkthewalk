import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader

from ttw.data_loader import TalkTheWalkEmergent, get_collate_fn
from ttw.utils import create_logger
from ttw.modules import MASC, NoMASC, CBoW

class TouristContinuous(nn.Module):

    def __init__(self, emb_sz, num_observations, T=2, apply_masc=True):
        super(TouristContinuous, self).__init__()
        self.num_embeddings = num_observations
        self.emb_sz = emb_sz
        self.apply_masc = apply_masc
        self.T = T
        self.goldstandard_emb = nn.Embedding(num_observations, emb_sz)

        self.obs_write_gate = nn.ParameterList()
        for _ in range(T + 1):
            self.obs_write_gate.append(nn.Parameter(torch.FloatTensor(1, emb_sz).normal_(0.0, 0.1)))

        if apply_masc:
            self.action_emb = nn.Embedding(4, emb_sz)
            self.act_write_gate = nn.Parameter(torch.FloatTensor(1, T, emb_sz).normal_(0.0, 0.1))


    def forward(self, batch):
        embs = list()
        for step in range(self.T + 1):
            emb = self.goldstandard_emb.forward(batch['goldstandard'][:, step, :]).sum(dim=1)
            emb = emb * F.sigmoid(self.obs_write_gate[step])
            embs.append(emb)
        obs_msg = sum(embs)

        act_msg = None
        if self.apply_masc:
            action_emb = self.action_emb.forward(batch['actions'])
            action_emb *= F.sigmoid(self.act_write_gate)
            act_msg = action_emb.sum(dim=1)

        return obs_msg, act_msg


class GuideContinuous(nn.Module):

    def __init__(self, emb_sz, num_embeddings, T=2, apply_masc=True):
        super(GuideContinuous, self).__init__()
        self.num_embeddings = num_embeddings
        self.emb_sz = emb_sz
        self.apply_masc = apply_masc
        self.T = T

        self.landmark_write_gate = nn.ParameterList()
        for _ in range(self.T + 1):
            self.landmark_write_gate.append(nn.Parameter(torch.FloatTensor(1, emb_sz, 1, 1).normal_(0.0, 0.1)))

        self.cbow_fn = CBoW(num_embeddings, emb_sz, init_std=0.01)

        if self.apply_masc:
            self.masc_fn = MASC(emb_sz)
            self.extract_fns = nn.ModuleList()
            for _ in range(T):
                self.extract_fns.append(nn.Linear(emb_sz, 9))
        else:
            self.masc_fn = NoMASC(emb_sz)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, msg, batch):
        obs_msg, act_msg = msg

        l_emb = self.cbow_fn.forward(batch['landmarks']).permute(0, 3, 1, 2)
        l_embs = [l_emb]

        if self.apply_masc:
            for j in range(self.T):
                act_mask = self.extract_fns[j](act_msg)
                out = self.masc_fn.forward(l_embs[-1], act_mask)
                l_embs.append(out)
        else:
            for j in range(self.T):
                out = self.masc_fn.forward(l_emb)
                l_embs.append(out)

        landmarks = sum([F.sigmoid(gate)*emb for gate, emb in zip(self.landmark_write_gate, l_embs)])
        landmarks = landmarks.resize(l_emb.size(0), landmarks.size(1), 16).transpose(1, 2)

        logits = torch.bmm(landmarks, obs_msg.unsqueeze(-1)).squeeze(-1)
        prob = F.softmax(logits, dim=1)

        y_true = (batch['target'][:, 0]*4 + batch['target'][:, 1]).squeeze(-1)

        loss = self.loss(prob, y_true)
        acc = sum([1.0 for pred, target in zip(prob.max(1)[1].data.cpu().numpy(), y_true.data.cpu().numpy()) if pred == target])/y_true.size(0)
        return loss, acc, prob


def epoch(loader, tourist, guide, opt=None):
    l, a = 0.0, 0.0
    n_batches = 0
    for batch in loader:
        msg = tourist.forward(batch)
        loss, acc, _ = guide.forward(msg, batch)

        l += loss.item()
        a += acc
        n_batches += 1

        if opt:
            for o in opt:
                o.zero_grad()
            loss.backward()
            for o in opt:
                o.step()
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

    t_opt = optim.Adam(tourist.parameters())
    g_opt = optim.Adam(guide.parameters())

    if args.cuda:
        tourist = tourist.cuda()
        guide = guide.cuda()

    best_train_loss, best_valid_loss, best_test_loss = None, 1e16, None
    best_train_acc, best_valid_acc, best_test_acc = None, None, None

    for i in range(1, args.num_epochs + 1):
        # train
        train_loss, train_acc = epoch(train_loader, tourist, guide, opt=[t_opt, g_opt])
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

            # tourist.save(os.path.join(exp_dir, 'tourist.pt'))
            # guide.save(os.path.join(exp_dir, 'guide.pt'))

    logger.info("%.2f, %.2f. %.2f" % (best_train_acc*100, best_valid_acc*100, best_test_acc*100))
