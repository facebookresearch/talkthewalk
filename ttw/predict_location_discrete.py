import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from ttw.data_loader import TalkTheWalkEmergent, get_collate_fn
from ttw.utils import create_logger
from ttw.modules import MASC, NoMASC, CBoW


class GuideDiscrete(nn.Module):
    def __init__(self, in_vocab_sz, num_landmarks, apply_masc=True, T=2):
        super(GuideDiscrete, self).__init__()
        self.in_vocab_sz = in_vocab_sz
        self.num_landmarks = num_landmarks
        self.T = T
        self.apply_masc = apply_masc
        self.emb_map = CBoW(num_landmarks, in_vocab_sz, init_std=0.1)
        self.obs_emb_fn = nn.Linear(in_vocab_sz, in_vocab_sz)
        self.landmark_write_gate = nn.ParameterList()
        for _ in range(T + 1):
            self.landmark_write_gate.append(nn.Parameter(torch.FloatTensor(1, in_vocab_sz, 1, 1).normal_(0.0, 0.1)))

        if apply_masc:
            self.masc_fn = MASC(in_vocab_sz)
            self.action_emb = nn.ModuleList()
            for i in range(T):
                self.action_emb.append(nn.Linear(in_vocab_sz, 9))
        else:
            self.masc_fn = NoMASC(in_vocab_sz)


    def forward(self, message, landmarks):
        msg_obs = self.obs_emb_fn(message[0])
        batch_size = message[0].size(0)

        landmark_emb = self.emb_map.forward(landmarks).permute(0, 3, 1, 2)
        landmark_embs = [landmark_emb]

        if self.apply_masc:
            for j in range(self.T):
                act_msg = message[1]
                action_out = self.action_emb[j](act_msg)
                out = self.masc_fn.forward(landmark_embs[-1], action_out, current_step=j)
                landmark_embs.append(out)
        else:
            for j in range(self.T):
                out = self.masc_fn.forward(landmark_embs[-1])
                landmark_embs.append(out)

        landmarks = sum([F.sigmoid(gate) * emb for gate, emb in zip(self.landmark_write_gate, landmark_embs)])
        landmarks = landmarks.view(batch_size, landmarks.size(1), 16).transpose(1, 2)

        logits = torch.bmm(landmarks, msg_obs.unsqueeze(-1)).squeeze(-1)
        prob = F.softmax(logits, dim=1)
        return prob

    def save(self, path):
        state = dict()
        state['in_vocab_sz'] = self.in_vocab_sz
        state['num_landmarks'] = self.num_landmarks
        state['parameters'] = self.state_dict()
        state['T'] = self.T
        state['apply_masc'] = self.apply_masc
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        guide = cls(state['in_vocab_sz'], state['num_landmarks'], T=state['T'],
                    apply_masc=state['apply_masc'])
        guide.load_state_dict(state['parameters'])
        return guide


class TouristDiscrete(nn.Module):
    def __init__(self, vocab_sz, num_observations, T=2, apply_masc=False):
        super(TouristDiscrete, self).__init__()
        self.T = T
        self.apply_masc = apply_masc
        self.vocab_sz = vocab_sz
        self.num_observations = num_observations

        self.goldstandard_emb = nn.Embedding(num_observations, vocab_sz)

        self.num_embeddings = T + 1
        self.obs_write_gate = nn.ParameterList()
        for _ in range(T + 1):
            self.obs_write_gate.append(nn.Parameter(torch.FloatTensor(1, vocab_sz).normal_(0.0, 0.1)))

        if self.apply_masc:
            self.action_emb = nn.Embedding(4, vocab_sz)
            self.num_embeddings += T
            self.act_write_gate = nn.ParameterList()
            for _ in range(T):
                self.act_write_gate.append(nn.Parameter(torch.FloatTensor(1, vocab_sz).normal_(0.0, 0.1)))

        self.loss = nn.CrossEntropyLoss()
        self.value_pred = nn.Linear((1 + int(self.apply_masc)) * self.vocab_sz, 1)

    def forward(self, batch, greedy=False):
        batch_size = batch['actions'].size(0)
        feat_emb = list()

        max_steps = self.T + 1
        for step in range(max_steps):
            emb = self.goldstandard_emb.forward(batch['goldstandard'][:, step, :]).sum(dim=1)
            emb = emb * F.sigmoid(self.obs_write_gate[step])
            feat_emb.append(emb)

        act_emb = list()
        if self.apply_masc:
            for step in range(self.T):
                emb = self.action_emb.forward(batch['actions'][:, step])
                emb = emb * F.sigmoid(self.act_write_gate[step])
                act_emb.append(emb)

        comms = list()
        probs = list()

        feat_embeddings = sum(feat_emb)
        feat_logits = feat_embeddings
        feat_prob = F.sigmoid(feat_logits).cpu()
        feat_msg = feat_prob.bernoulli().detach()

        probs.append(feat_prob)
        comms.append(feat_msg)

        if self.apply_masc:
            act_embeddings = sum(act_emb)
            act_logits = act_embeddings
            act_prob = F.sigmoid(act_logits).cpu()
            act_msg = act_prob.bernoulli().detach()

            probs.append(act_prob)
            comms.append(act_msg)

        if self.apply_masc:
            embeddings = torch.cat([feat_embeddings, act_embeddings], 1).resize(batch_size, 2 * self.vocab_sz)
        else:
            embeddings = feat_embeddings
        value = self.value_pred(embeddings)

        return comms, probs, value

    def save(self, path):
        state = dict()
        state['num_observations'] = self.vocab_sz
        state['vocab_sz'] = self.vocab_sz
        state['T'] = self.T
        state['apply_masc'] = self.apply_masc
        state['parameters'] = self.state_dict()
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        tourist = cls(state['vocab_sz'], state['num_observations'], T=state['T'],
                      apply_masc=state['apply_masc'])
        tourist.load_state_dict(state['parameters'])
        return tourist

def eval_epoch(loader, tourist, guide, cuda, t_opt=None, g_opt=None):
    tourist.eval()
    guide.eval()

    correct, total = 0, 0
    for batch in loader:
        # forward
        t_comms, t_probs, t_val = tourist(batch)
        if cuda:
            t_comms = [x.cuda() for x in t_comms]
        out_g = guide(t_comms, batch['landmarks'])

        # acc
        tgt = (batch['target'][:, 0] * 4 + batch['target'][:, 1])
        pred = torch.max(out_g, 1)[1]
        correct += sum(
            [1.0 for y_hat, y_true in zip(pred, tgt) if y_hat == y_true])
        total += len(batch['target'])

        if t_opt and g_opt:
            # train if optimizers are specified
            g_loss = -torch.log(torch.gather(out_g, 1, tgt.unsqueeze(-1)))
            _, max_ind = torch.max(out_g, 1)

            # tourist loss
            rewards = -g_loss  # tourist reward is log likelihood of correct answer

            t_rl_loss = 0.
            eps = 1e-16

            advantage = Variable((rewards.data - t_val.data))
            if cuda:
                advantage = advantage.cuda()
            t_val_loss = ((t_val - Variable(rewards.data)) ** 2).mean()  # mse

            for action, prob in zip(t_comms, t_probs):
                if args.cuda:
                    action = action.cuda()
                    prob = prob.cuda()
                action_prob = action * prob + (1.0 - action) * (1.0 - prob)

                t_rl_loss -= (torch.log(action_prob + eps) * advantage).sum()

            # backward
            g_opt.zero_grad()
            t_opt.zero_grad()
            g_loss.mean().backward()
            (t_rl_loss + t_val_loss).backward()
            torch.nn.utils.clip_grad_norm(tourist.parameters(), 5)
            torch.nn.utils.clip_grad_norm(guide.parameters(), 5)
            g_opt.step()
            t_opt.step()

    return correct / total



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
    tourist = TouristDiscrete(args.vocab_sz, len(train_data.map.landmark_dict),
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
