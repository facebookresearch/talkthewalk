import argparse
import os
import json
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.dataloader import DataLoader

from ttw.data_loader import TalkTheWalkLanguage, get_collate_fn
from ttw.modules import CBoW, MASC, NoMASC, ControlStep
from ttw.utils import create_logger


class Guide(nn.Module):

    def __init__(self, inp_emb_sz, hidden_sz, num_tokens, apply_masc=True, T=1):
        super(Guide, self).__init__()
        self.hidden_sz = hidden_sz
        self.inp_emb_sz = inp_emb_sz
        self.num_tokens = num_tokens
        self.apply_masc = apply_masc
        self.T = T

        self.embed_fn = nn.Embedding(num_tokens, inp_emb_sz, padding_idx=0)
        self.encoder_fn = nn.LSTM(inp_emb_sz, hidden_sz//2, batch_first=True, bidirectional=True)
        self.cbow_fn = CBoW(11, hidden_sz)

        self.T_prediction_fn = nn.Linear(hidden_sz, T+1)

        self.feat_control_emb = nn.Parameter(torch.FloatTensor(hidden_sz).normal_(0.0, 0.1))
        self.feat_control_step_fn = ControlStep(hidden_sz)

        if apply_masc:
            self.act_control_emb = nn.Parameter(torch.FloatTensor(hidden_sz).normal_(0.0, 0.1))
            self.act_control_step_fn = ControlStep(hidden_sz)
            self.action_linear_fn = nn.Linear(hidden_sz, 9)

        self.landmark_write_gate = nn.ParameterList()
        self.obs_write_gate = nn.ParameterList()
        for _ in range(T + 1):
            self.landmark_write_gate.append(nn.Parameter(torch.FloatTensor(1, hidden_sz, 1, 1).normal_(0, 0.1)))
            self.obs_write_gate.append(nn.Parameter(torch.FloatTensor(1, hidden_sz).normal_(0.0, 0.1)))

        if apply_masc:
            self.masc_fn = MASC(self.hidden_sz)
        else:
            self.masc_fn = NoMASC(self.hidden_sz)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch, add_rl_loss=False):
        batch_size = batch['utterance'].size(0)
        input_emb = self.embed_fn(batch['utterance'])
        hidden_states, _ = self.encoder_fn(input_emb)

        last_state_indices = batch['utterance_mask'].sum(1).long() - 1

        last_hidden_states = hidden_states[torch.arange(batch_size).long(), last_state_indices, :]
        T_dist = F.softmax(self.T_prediction_fn(last_hidden_states))
        sampled_Ts = T_dist.multinomial(1).squeeze()

        obs_msgs = list()
        feat_controller = self.feat_control_emb.unsqueeze(0).repeat(batch_size, 1)
        for step in range(self.T + 1):
            extracted_msg, feat_controller = self.feat_control_step_fn(hidden_states, batch['utterance_mask'], feat_controller)
            obs_msgs.append(extracted_msg)

        tourist_obs_msg = []
        for i, (gate, emb) in enumerate(zip(self.obs_write_gate, obs_msgs)):
            include = (i < sampled_Ts).float().unsqueeze(-1)
            tourist_obs_msg.append(include*F.sigmoid(gate)*emb)
        tourist_obs_msg = sum(tourist_obs_msg)


        landmark_emb = self.cbow_fn(batch['landmarks']).permute(0, 3, 1, 2)
        landmark_embs = [landmark_emb]

        if self.apply_masc:
            act_controller = self.act_control_emb.unsqueeze(0).repeat(batch_size, 1)
            for step in range(self.T):
                extracted_msg, act_controller = self.act_control_step_fn(hidden_states, batch['utterance_mask'], act_controller)
                action_out = self.action_linear_fn(extracted_msg)
                out = self.masc_fn.forward(landmark_embs[-1], action_out, current_step=step, Ts=sampled_Ts)
                landmark_embs.append(out)
        else:
            for step in range(self.T):
                landmark_embs.append(self.masc_fn.forward(landmark_embs[-1]))

        landmarks = sum([F.sigmoid(gate)*emb for gate, emb in zip(self.landmark_write_gate, landmark_embs)])

        landmarks = landmarks.resize(batch_size, landmarks.size(1), 16).transpose(1, 2)

        logits = torch.bmm(landmarks, tourist_obs_msg.unsqueeze(-1)).squeeze(-1)
        prob = F.softmax(logits, dim=1)
        y_true = (batch['target'][:, 0] * 4 + batch['target'][:, 1]).squeeze()

        sl_loss = -torch.log(torch.gather(prob, 1, y_true.unsqueeze(-1)) + 1e-8)

        # add RL loss
        loss = sl_loss
        if add_rl_loss:
            reward = -(sl_loss - sl_loss.mean())

            log_prob = torch.log(torch.gather(T_dist, 1, sampled_Ts.unsqueeze(-1)) + 1e-8)
            rl_loss = (log_prob*reward)
            loss = loss - rl_loss

        acc = sum([1.0 for pred, target in zip(prob.max(1)[1].data.cpu().numpy(), y_true.data.cpu().numpy()) if
                   pred == target]) / batch_size
        return loss, acc

    def save(self, path):
        state = dict()
        state['hidden_sz'] = self.hidden_sz
        state['embed_sz'] = self.inp_emb_sz
        state['num_tokens'] = self.num_tokens
        state['apply_masc'] = self.apply_masc
        state['T'] = self.T
        state['parameters'] = self.state_dict()
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        guide = cls(state['embed_sz'], state['hidden_sz'], state['num_tokens'],
                    T=state['T'], apply_masc=state['apply_masc'])
        guide.load_state_dict(state['parameters'])
        return guide

def eval_epoch(loader, guide, opt=None):
    loss, accs, total = 0.0, 0.0, 0.0

    for batch in loader:
        l, acc = guide.forward(batch, add_rl_loss=True)
        accs += acc
        total += 1
        l = l.sum()
        loss += l.cpu().data.numpy()

        if opt is not None:
            opt.zero_grad()
            l.backward()
            opt.step()
    return loss/total, accs/total

def get_mean_T(guide, Xs, landmarks, ys, cuda=True, batch_sz=64):
    distribution = numpy.array([0.0] * 4)
    for jj in range(0, len(Xs), batch_sz):
        X_batch, mask, landmark_batch, y_batch = create_batch(Xs[jj:jj + batch_sz],
                                                              landmarks[jj:jj + batch_sz],
                                                              ys[jj:jj + batch_sz], cuda=cuda)
        batch_sz = X_batch.size(0)

        input_emb = guide.embed_fn(X_batch)
        hidden_states, _ = guide.encoder_fn(input_emb)

        last_state_indices = mask.sum(1).long() - 1
        last_hidden_states = hidden_states[torch.arange(batch_sz).long(), last_state_indices, :]
        T_dist = F.softmax(guide.T_prediction_fn(last_hidden_states))

        distribution += T_dist.sum(0).cpu().data.numpy()
    distribution /= len(Xs)
    mean_T = sum([p*v for p, v in zip(distribution, range(len(distribution)))])
    return mean_T



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--exp-dir', type=str, default='./exp')
    parser.add_argument('--apply-masc', action='store_true')
    parser.add_argument('--T', type=int, default=3)
    parser.add_argument('--hidden-sz', type=int, default=256)
    parser.add_argument('--embed-sz', type=int, default=128)
    parser.add_argument('--last-turns', type=int, default=1)
    parser.add_argument('--batch-sz', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--exp-name', type=str, default='test')

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


    guide = Guide(args.embed_sz, args.hidden_sz, len(train_data.dict), apply_masc=args.apply_masc, T=args.T)

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

    # best_guide = Guide.load(os.path.join(exp_dir, 'guide.pt'))
    # if args.cuda:
    #     best_guide = best_guide.cuda()
    # logger.info("mean T: {}".format(get_mean_T(best_guide, test_Xs, test_landmarks, test_ys)))
