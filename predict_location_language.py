import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from sklearn.utils import shuffle
from talkthewalk.data_loader import Landmarks, step_aware, to_variable
from talkthewalk.modules import CBoW, MASC, NoMASC, ControlStep
from talkthewalk.utils import create_logger
from talkthewalk.dict import Dictionary

def get_action(msg):
    msg_to_act = {'ACTION:TURNLEFT': 0, 'ACTION:TURNRIGHT': 1, 'ACTION:FORWARD': 2}
    return msg_to_act.get(msg, None)

def load_data(dataset, landmark_map, dictionary, last_turns=1):
    Xs = list()
    landmarks = list()
    ys = list()
    for config in dataset:
        loc = config['start_location']
        boundaries = config['boundaries']

        dialogue = list()
        for msg in config['dialog']:
            if msg['id'] == 'Tourist':
                act = get_action(msg['text'])
                if act is not None:
                    loc = step_aware(act, loc, boundaries)
                elif len(msg['text'].split(' ')) > 2:
                    dialogue.append(dictionary.encode(msg['text']))
                    utt = [y for x in dialogue[-last_turns:] for y in x]
                    Xs.append(utt)

                    ls, y = landmark_map.get_landmarks_2d(config['neighborhood'], boundaries, loc)
                    landmarks.append(ls)
                    ys.append(y)
            else:
                dialogue.append(dictionary.encode(msg['text']))

    return Xs, landmarks, ys

def create_batch(Xs, landmarks, ys, cuda=False):
    batch_size = len(Xs)
    seq_lens = [len(seq) for seq in Xs]
    max_len = max(seq_lens)

    X_batch = torch.LongTensor(batch_size, max_len).zero_()
    mask = torch.FloatTensor(batch_size, max_len).zero_()
    for i, seq in enumerate(Xs):
        X_batch[i, :len(seq)] = torch.LongTensor(seq)
        mask[i, :len(seq)] = 1.0

    max_landmarks_per_coord = max([max([max([len(y) for y in x]) for x in l]) for l in landmarks])
    landmark_batch = torch.LongTensor(batch_size, 4, 4, max_landmarks_per_coord).zero_()

    for i, ls in enumerate(landmarks):
        for j in range(4):
            for k in range(4):
                landmark_batch[i, j, k, :len(landmarks[i][j][k])] = torch.LongTensor(landmarks[i][j][k])

    y_batch = torch.LongTensor(ys)

    return to_variable([X_batch, mask, landmark_batch, y_batch], cuda=cuda)


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

    def forward(self, Xs, seq_mask, landmarks, ys):
        batch_size = Xs.size(0)
        input_emb = self.embed_fn(Xs)
        hidden_states, _ = self.encoder_fn(input_emb)

        last_state_indices = seq_mask.sum(1).long() - 1
        last_hidden_states = hidden_states[torch.arange(batch_size).long(), last_state_indices, :]
        T_dist = F.softmax(self.T_prediction_fn(last_hidden_states))
        sampled_Ts = T_dist.multinomial(1).squeeze()

        obs_msgs = list()
        feat_controller = self.feat_control_emb.unsqueeze(0).repeat(batch_size, 1)
        for step in range(self.T + 1):
            extracted_msg, feat_controller = self.feat_control_step_fn(hidden_states, seq_mask, feat_controller)
            obs_msgs.append(extracted_msg)

        tourist_obs_msg = []
        for i, (gate, emb) in enumerate(zip(self.obs_write_gate, obs_msgs)):
            include = (i < sampled_Ts).float().unsqueeze(-1)
            tourist_obs_msg.append(include*F.sigmoid(gate)*emb)
        tourist_obs_msg = sum(tourist_obs_msg)


        landmark_emb = self.cbow_fn(landmarks).permute(0, 3, 1, 2)
        landmark_embs = [landmark_emb]

        if self.apply_masc:
            act_controller = self.act_control_emb.unsqueeze(0).repeat(batch_size, 1)
            for step in range(self.T):
                extracted_msg, act_controller = self.act_control_step_fn(hidden_states, seq_mask, act_controller)
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
        y_true = (ys[:, 0] * 4 + ys[:, 1]).squeeze()

        sl_loss = -torch.log(torch.gather(prob, 1, y_true.unsqueeze(-1)) + 1e-8)
        # add RL loss
        reward = -(sl_loss - sl_loss.mean())

        log_prob = torch.log(torch.gather(T_dist, 1, sampled_Ts.unsqueeze(-1)) + 1e-8)
        rl_loss = (log_prob*reward).sum()
        loss = sl_loss.sum() - rl_loss

        acc = sum([1.0 for pred, target in zip(prob.max(1)[1].data.cpu().numpy(), y_true.data.cpu().numpy()) if
                   pred == target]) / batch_size
        return loss, acc

    def save(self, path):
        print('Save to: ' + path)
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

def eval_epoch(net, Xs, landmarks, ys, batch_sz, opt=None, cuda=False):
    loss, accs, total = 0.0, 0.0, 0.0

    for jj in range(0, len(Xs), batch_sz):
        X_batch, mask, landmark_batch, y_batch = create_batch(Xs[jj:jj + batch_sz], landmarks[jj:jj + batch_sz],
                                                              ys[jj:jj + batch_sz], cuda=cuda)
        l, acc = net.forward(X_batch, mask, landmark_batch, y_batch)
        accs += acc
        total += 1
        loss += l.cpu().data.numpy()

        if opt is not None:
            opt.zero_grad()
            l.backward()
            opt.step()
    return loss/total, accs/total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--last-turns', type=int, default=1)
    parser.add_argument('--masc', action='store_true')
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--hidden-sz', type=int, default=256)
    parser.add_argument('--embed-sz', type=int, default=128)
    parser.add_argument('--batch-sz', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--exp-name', type=str, default='test')

    args = parser.parse_args()

    exp_dir = os.path.join(os.environ['TALKTHEWALK_EXPDIR'], args.exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    logger = create_logger(os.path.join(exp_dir, 'log.txt'))
    logger.info(args)

    data_dir = os.environ.get('TALKTHEWALK_DATADIR', './data')

    train_set = json.load(open(os.path.join(data_dir, 'talkthewalk.train.json')))
    valid_set = json.load(open(os.path.join(data_dir, 'talkthewalk.valid.json')))
    test_set = json.load(open(os.path.join(data_dir, 'talkthewalk.test.json')))

    dictionary = Dictionary('./data/dict.txt', 3)

    neighborhoods = ['fidi', 'hellskitchen', 'williamsburg', 'uppereast', 'eastvillage']
    landmark_map = Landmarks(neighborhoods, include_empty_corners=True)

    train_Xs, train_landmarks, train_ys = load_data(train_set, landmark_map, dictionary, last_turns=args.last_turns)
    valid_Xs, valid_landmarks, valid_ys = load_data(valid_set, landmark_map, dictionary, last_turns=args.last_turns)
    test_Xs, test_landmarks, test_ys = load_data(test_set, landmark_map, dictionary, last_turns=args.last_turns)

    net = Guide(args.embed_sz, args.hidden_sz, len(dictionary), apply_masc=args.masc, T=args.T)

    if args.cuda:
        net = net.cuda()
    opt = optim.Adam(net.parameters())

    best_train_acc, best_val_acc, best_test_acc = 0.0, 0.0, 0.0
    for i in range(args.num_epochs):
        train_Xs, train_landmarks, train_ys = shuffle(train_Xs, train_landmarks, train_ys)
        train_loss, train_acc = eval_epoch(net, train_Xs, train_landmarks, train_ys, args.batch_sz, opt=opt, cuda=args.cuda)
        valid_loss, valid_acc = eval_epoch(net, valid_Xs, valid_landmarks, valid_ys, args.batch_sz, cuda=args.cuda)
        test_loss, test_acc = eval_epoch(net, test_Xs, test_landmarks, test_ys, args.batch_sz, cuda=args.cuda)

        logger.info("Train loss: %.2f, Valid loss: %.2f, Test loss: %.2f" % (train_loss, valid_loss, test_loss))
        logger.info("Train acc: %.2f, Valid acc: %.2f, Test acc: %.2f" % (train_acc*100, valid_acc*100, test_acc*100))

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_train_acc, best_val_acc, best_test_acc = train_acc, valid_acc, test_acc
            net.save(os.path.join(exp_dir, 'guide.pt'))

    logger.info(best_train_acc)
    logger.info(best_val_acc)
    logger.info(best_test_acc)