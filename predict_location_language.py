import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from data_loader import Landmarks, step_aware, to_variable
from utils import create_logger
from dict import Dictionary
from predict_location_multiple_step import MapEmbedding2d

def get_action(msg):
    msg_to_act = {'ACTION:TURNLEFT': 0, 'ACTION:TURNRIGHT': 1, 'ACTION:FORWARD': 2}
    return msg_to_act.get(msg, None)

def load_data(dataset, landmark_map, dictionary, full_dialogue=False):
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
                    if full_dialogue:
                        dialogue.extend(dictionary.encode(msg['text']))
                        Xs.append(dialogue)
                    else:
                        Xs.append(dictionary.encode(msg['text']))
                    ls, y = landmark_map.get_landmarks_2d(config['neighborhood'], boundaries, loc)
                    landmarks.append(ls)
                    ys.append(y)
            elif full_dialogue:
                dialogue.extend(dictionary.encode(msg['text']))

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


class LocationPredictor(nn.Module):

    def __init__(self, inp_emb_sz, hidden_sz, num_tokens, max_steps=1, condition_on_action=False, mask_conv=True):
        super(LocationPredictor, self).__init__()
        self.hidden_sz = hidden_sz
        self.inp_emb_sz = inp_emb_sz
        self.num_tokens = num_tokens
        self.embedder = nn.Embedding(num_tokens, inp_emb_sz, padding_idx=0)
        self.encoder = nn.LSTM(inp_emb_sz, hidden_sz//2, batch_first=True, bidirectional=True)
        self.emb_map = MapEmbedding2d(11, hidden_sz)
        self.condition_on_action = condition_on_action
        self.mask_conv = mask_conv
        self.feat_control_emb = Parameter(torch.FloatTensor(hidden_sz).normal_(0.0, 0.1))
        self.feat_control_updater = nn.Linear(2*hidden_sz, hidden_sz)
        if condition_on_action:
            self.act_control_emb = Parameter(torch.FloatTensor(hidden_sz).normal_(0.0, 0.1))
            self.action_lin = nn.Linear(hidden_sz, 9)
            self.act_control_updater = nn.Linear(2*hidden_sz, hidden_sz)

        self.conv_weight = nn.Parameter(torch.FloatTensor(
            hidden_sz, hidden_sz, 3, 3))
        std = 1.0 / (hidden_sz * 9)
        self.conv_weight.data.uniform_(-std, std)
        self.max_steps = max_steps
        self.loss = nn.CrossEntropyLoss()

    def forward(self, Xs, seq_mask, landmarks, ys):
        batch_size = Xs.size(0)
        input_emb = self.embedder.forward(Xs)
        hidden_states, _ = self.encoder.forward(input_emb)

        feature_msgs = list()
        controller = self.feat_control_emb.unsqueeze(0).repeat(batch_size, 1)
        for step in range(self.max_steps):
            score = torch.bmm(hidden_states, controller.unsqueeze(-1)).squeeze(-1)
            score = score - 1e30*(1.0-seq_mask)
            att_score = F.softmax(score, dim=-1)
            extracted_msg = torch.bmm(att_score.unsqueeze(1), hidden_states).squeeze()
            feature_msgs.append(extracted_msg)
            controller = self.feat_control_updater.forward(torch.cat([extracted_msg, controller], 1))

        feature_msg = torch.cat(feature_msgs, 1)

        l_emb = self.emb_map.forward(landmarks).permute(0, 3, 1, 2)
        l_embs = [l_emb]

        if self.condition_on_action:
            act_controller = self.act_control_emb.unsqueeze(0).repeat(batch_size, 1)
            for step in range(self.max_steps-1):
                score = torch.bmm(hidden_states, act_controller.unsqueeze(-1)).squeeze(-1)
                score = score - 1e30*(1-seq_mask)
                att_score = F.softmax(score, dim=-1)
                extracted_msg = torch.bmm(att_score.unsqueeze(1), hidden_states).squeeze()
                act_controller = self.act_control_updater.forward(torch.cat([extracted_msg, act_controller], 1))
                action_out = self.action_lin.forward(extracted_msg)
                out = Variable(torch.FloatTensor(batch_size, self.hidden_sz, 4, 4).zero_().cuda())
                for i in range(batch_size):
                    selected_inp = l_embs[-1][i, :, :, :].unsqueeze(0)
                    mask = F.softmax(action_out[i], dim=0).resize(1, 1, 3, 3)
                    weight = mask * self.conv_weight
                    out[i, :, :, :] = F.conv2d(selected_inp, weight, padding=1).squeeze(0)
                l_embs.append(out)
        else:
            weight = self.conv_weight
            if self.mask_conv:
                mask = torch.FloatTensor(1, 1, 3, 3).cuda().zero_()
                mask[0, 0, 0, 1] = 1.0
                mask[0, 0, 1, 0] = 1.0
                mask[0, 0, 2, 1] = 1.0
                mask[0, 0, 1, 2] = 1.0
                weight = self.conv_weight * Variable(mask)
            for j in range(self.max_steps - 1):
                tmp = F.conv2d(l_embs[-1], weight, padding=1)
                l_embs.append(tmp)

        landmarks = torch.cat(l_embs, 1)
        landmarks = landmarks.resize(batch_size, landmarks.size(1), 16).transpose(1, 2)

        logits = torch.bmm(landmarks, feature_msg.unsqueeze(-1)).squeeze(-1)
        prob = F.softmax(logits, dim=1)
        y_true = (ys[:, 0] * 4 + ys[:, 1]).squeeze()
        loss = self.loss(prob, y_true)
        acc = sum([1.0 for pred, target in zip(prob.max(1)[1].data.cpu().numpy(), y_true.data.cpu().numpy()) if
                   pred == target]) / batch_size
        return loss, acc

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
    parser.add_argument('--full-dialogue', action='store_true')
    parser.add_argument('--condition-on-action', action='store_true')
    parser.add_argument('--mask-conv', action='store_true')
    parser.add_argument('--num-steps', type=int, default=2)
    parser.add_argument('--hidden-sz', type=int, default=256)
    parser.add_argument('--embed-sz', type=int, default=128)
    parser.add_argument('--batch-sz', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--exp-name', type=str, default='test')

    args = parser.parse_args()

    exp_dir = os.path.join(os.environ['TALKTHEWALK_EXPDIR'], args.exp_name)
    if os.path.exists(exp_dir):
        raise RuntimeError('Experiment directory already exist..')
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

    train_Xs, train_landmarks, train_ys = load_data(train_set, landmark_map, dictionary, full_dialogue=args.full_dialogue)
    valid_Xs, valid_landmarks, valid_ys = load_data(valid_set, landmark_map, dictionary, full_dialogue=args.full_dialogue)
    test_Xs, test_landmarks, test_ys = load_data(test_set, landmark_map, dictionary, full_dialogue=args.full_dialogue)

    batch_sz = args.batch_sz
    hid_sz = 256
    emb_sz = 128
    cuda = True
    num_epochs = args.num_epochs

    net = LocationPredictor(emb_sz, hid_sz, len(dictionary), condition_on_action=args.condition_on_action,
                            mask_conv=args.mask_conv, max_steps=args.num_steps)

    if cuda:
        net = net.cuda()
    opt = optim.Adam(net.parameters())

    best_train_acc, best_val_acc, best_test_acc = 0.0, 0.0, 0.0
    for i in range(num_epochs):
        train_Xs, train_landmarks, train_ys = shuffle(train_Xs, train_landmarks, train_ys)
        train_loss, train_acc = eval_epoch(net, train_Xs, train_landmarks, train_ys, batch_sz, opt=opt, cuda=cuda)
        valid_loss, valid_acc = eval_epoch(net, valid_Xs, valid_landmarks, valid_ys, batch_sz, cuda=cuda)
        test_loss, test_acc = eval_epoch(net, test_Xs, test_landmarks, test_ys, batch_sz, cuda=cuda)

        logger.info("Train loss: %.2f, Valid loss: %.2f, Test loss: %.2f" % (train_loss, valid_loss, test_loss))
        logger.info("Train acc: %.2f, Valid acc: %.2f, Test acc: %.2f" % (train_acc*100, valid_acc*100, test_acc*100))

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_train_acc, best_val_acc, best_test_acc = train_acc, valid_acc, test_acc

    logger.info(best_train_acc)
    logger.info(best_val_acc)
    logger.info(best_test_acc)
