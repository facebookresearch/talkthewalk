import os
import json
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

from talkthewalk.data_loader import step_aware, GoldstandardFeatures, Landmarks, neighborhoods, to_variable
from talkthewalk.dict import Dictionary, START_TOKEN, END_TOKEN
from talkthewalk.modules import CBoW
from talkthewalk.utils import create_logger


def get_action(msg):
    msg_to_act = {'ACTION:TURNLEFT': 1,
                  'ACTION:TURNRIGHT': 2,
                  'ACTION:FORWARD': 3}
    return msg_to_act.get(msg, None)


# Determine if tourist went "up", "down", "left", "right"
def get_new_action(old_loc, new_loc):
    act_to_idx = {'LEFT': 1, 'UP': 2, 'RIGHT': 3, 'DOWN': 4, 'STAYED': 5}
    step_to_dir = {
        0: {
            1: 'N',
            -1: 'S',
            0: 'STAYED'
        },
        1: {
            0: 'E',
        },
        -1: {
            0: 'W'
        }
    }
    dir_to_act = {'N': 'UP', 'E': 'RIGHT', 'S': 'DOWN', 'W': 'LEFT', 'STAYED': 'STAYED'}

    step = [new_loc[0] - old_loc[0], new_loc[1] - old_loc[1]]
    direction = step_to_dir[step[0]][step[1]]
    return act_to_idx[dir_to_act[direction]]


def load_data(data, dictionary, feature_loader, landmark_map,
              orientation_aware=False,
              min_sent_length=3):
    observations = []  # x_i = [a_1, o_1, a_2, ..., a_n, o_n] acts + obs
    actions = []
    target_indices = []
    landmarks = []
    msgs = []

    for config in data:
        loc = config['start_location']
        boundaries = config['boundaries']
        neighborhood = config['neighborhood']
        obs_memory = list()
        obs_memory.append(feature_loader.get(neighborhood, loc))
        act_memory = list()

        for msg in config['dialog']:
            if msg['id'] == 'Tourist':
                act = get_action(msg['text'])
                if act is None:
                    msg_length = len(msg['text'].split(' '))
                    if msg_length > min_sent_length:
                        encoded_msg = [dictionary.tok2i[START_TOKEN]] + dictionary.encode(msg['text'], include_end=True)
                        msgs.append(encoded_msg)

                        ls, tl = landmark_map.get_landmarks_2d(
                            neighborhood, boundaries, loc)
                        landmarks.append(ls)
                        target_indices.append(tl)
                        observations.append(obs_memory)
                        actions.append(act_memory)
                        obs_memory = list()
                        obs_memory.append(feature_loader.get(neighborhood, loc))
                        act_memory = list()

                else:
                    new_loc = step_aware(act - 1, loc, boundaries)
                    old_loc = loc
                    loc = new_loc

                    if orientation_aware:
                        act_memory.append(act)
                        obs_memory.append(feature_loader.get(neighborhood, new_loc))
                    else:
                        if act == 3:  # went forward
                            act_dir = get_new_action(old_loc, new_loc)
                            act_memory.append(act_dir)
                            obs_memory.append(loader.get(neighborhood, loc))

    return observations, actions, msgs, landmarks, target_indices


def create_batch(observations, actions, messages, cuda=True):
    batch_sz = len(observations)
    obs_seq_len = [len(s) for s in observations]
    obs_seqlen_tensor = torch.LongTensor(obs_seq_len)
    max_steps = max(obs_seq_len)
    max_len = max([max([len(x) for x in l]) for l in observations])
    obs_tensor = torch.LongTensor(batch_sz, max_steps, max_len).zero_()
    for ii in range(batch_sz):
        for jj in range(len(observations[ii])):
            for kk in range(len(observations[ii][jj])):
                obs_tensor[ii, jj, kk] = observations[ii][jj][kk]

    act_seqlen_tensor = torch.LongTensor([len(s) for s in actions])
    max_len = max([len(s) for s in actions])
    act_tensor = torch.LongTensor(batch_sz, max_len).zero_()
    for ii in range(batch_sz):
        for jj in range(len(actions[ii])):
            act_tensor[ii][jj] = actions[ii][jj]

    max_msg_len = max([len(s) for s in messages])
    msg_tensor = torch.LongTensor(batch_sz, max_msg_len).zero_()
    mask_tensor = torch.FloatTensor(batch_sz, max_msg_len).zero_()
    for ii in range(batch_sz):
        for jj in range(len(messages[ii])):
            msg_tensor[ii, jj] = messages[ii][jj]
        mask_tensor[ii, :len(messages[ii])] = 1.0

    return to_variable([obs_tensor, obs_seqlen_tensor, act_tensor, act_seqlen_tensor, msg_tensor, mask_tensor], cuda=cuda)


class Tourist(nn.Module):

    def __init__(self, act_emb_sz, act_hid_sz, num_actions, obs_emb_sz, obs_hid_sz, num_observations,
                 decoder_emb_sz, decoder_hid_sz, num_words, start_token=0):
        super(Tourist, self).__init__()
        self.act_emb_sz = act_emb_sz
        self.act_hid_sz = act_hid_sz
        self.num_actions = num_actions

        self.obs_emb_sz = obs_emb_sz
        self.obs_hid_sz = obs_hid_sz
        self.num_observations = num_observations

        self.decoder_emb_sz = decoder_emb_sz
        self.decoder_hid_sz = decoder_hid_sz
        self.num_words = num_words

        self.act_emb_fn = nn.Embedding(num_actions, act_emb_sz)
        self.act_encoder = nn.GRU(act_emb_sz, act_hid_sz, batch_first=True)
        self.act_hidden = nn.Parameter(torch.FloatTensor(1, act_hid_sz).normal_(0.0, 0.1))

        self.obs_emb_fn = CBoW(num_observations, obs_emb_sz, init_std=0.1)
        self.obs_encoder = nn.GRU(obs_emb_sz, obs_hid_sz, batch_first=True)
        self.obs_hidden = nn.Parameter(torch.FloatTensor(1, obs_hid_sz).normal_(0.0, 0.1))

        self.emb_fn = nn.Embedding(num_words, decoder_emb_sz)
        self.emb_fn.weight.data.normal_(0.0, 0.1)
        self.decoder = nn.GRU(2*decoder_emb_sz, decoder_hid_sz, batch_first=True)

        self.context_linear = nn.Linear(act_hid_sz+obs_hid_sz, decoder_emb_sz)
        self.out_linear = nn.Linear(decoder_hid_sz, num_words)

        self.loss = nn.CrossEntropyLoss(reduce=False)
        self.start_token = start_token
        print(start_token)

    def pick_hidden_state(self, states, seq_lens):
        batch_size = seq_lens.size(0)
        return states[torch.arange(batch_size).long(), seq_lens - 1, :]

    def forward(self, observations, obs_seq_len, actions, act_seq_len, gt_messages=None, gt_mask=None, sample=True, max_sample_length=15):
        batch_size = observations.size(0)

        obs_inp_emb = self.obs_emb_fn(observations)
        obs_h, _ = self.obs_encoder(obs_inp_emb)
        observation_emb = self.pick_hidden_state(obs_h, obs_seq_len)

        act_inp_emb = self.act_emb_fn(actions)
        action_embs, _ = self.act_encoder(act_inp_emb)
        action_emb = self.pick_hidden_state(action_embs, act_seq_len)

        context_emb = torch.cat([observation_emb, action_emb], 1)
        context_emb = self.context_linear.forward(context_emb)

        if not sample:
            # teacher forcing
            assert(gt_mask is not None and gt_messages is not None)
            inp = gt_messages[:, :-1]
            tgt = gt_messages[:, 1:]

            # print(gt_messages[torch.arange(batch_size).long(), gt_mask.sum(1).long() - 1])
            # print(gt_messages[:, 0])

            inp_emb = self.emb_fn.forward(inp)

            # concatenate external emb

            context_emb = context_emb.view(batch_size, 1, self.decoder_emb_sz).repeat(1, inp_emb.size(1), 1)
            inp_emb = torch.cat([inp_emb, context_emb], 2)

            hs, _ = self.decoder(inp_emb)

            score = self.out_linear(hs)

            loss = 0.0
            mask = gt_mask[:, 1:]

            for j in range(score.size(1)):
                flat_mask = mask[:, j]
                flat_score = score[:, j, :]
                flat_tgt = tgt[:, j]
                nll = self.loss(flat_score, flat_tgt)
                loss += (flat_mask*nll).sum()

            out = {}
            out['loss'] = loss
        else:
            input_ind = torch.LongTensor([self.start_token]*batch_size)
            if actions.is_cuda:
                input_ind = input_ind.cuda()

            preds = []
            probs = []

            hs = Variable(torch.FloatTensor(1, batch_size, self.decoder_hid_sz).fill_(0.0))
            if observations.is_cuda:
                hs = hs.cuda()

            for k in range(max_sample_length):
                inp_emb = self.emb_fn.forward(input_ind.unsqueeze(-1))

                context_emb = context_emb.view(batch_size, 1, self.decoder_emb_sz).repeat(1, inp_emb.size(1), 1)
                inp_emb = torch.cat([inp_emb, context_emb], 2)

                _, hs = self.decoder(inp_emb, hs)

                prob = F.softmax(self.out_linear(hs.squeeze(0)), dim=-1)
                samples = prob.multinomial(1)

                preds.append(samples)
                probs.append(prob)
                input_ind = samples.squeeze()

            out = {}
            out['preds'] = torch.cat(preds, 1)

        return out

    def save(self, path):
        print('Save to: ' + path)
        state = dict()
        state['act_emb_sz'] = self.act_emb_sz
        state['act_hid_sz'] = self.act_hid_sz
        state['num_actions'] = self.num_actions
        state['obs_emb_sz'] = self.obs_emb_sz
        state['obs_hid_sz'] = self.obs_hid_sz
        state['num_observations'] = self.num_observations
        state['decoder_emb_sz'] = self.decoder_emb_sz
        state['decoder_hid_sz'] = self.decoder_hid_sz
        state['num_words'] = self.num_words
        state['start_token'] = self.start_token
        state['parameters'] = self.state_dict()
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)

        tourist = cls(state['act_emb_sz'], state['act_hid_sz'], state['num_actions'],
                      state['obs_emb_sz'], state['obs_hid_sz'], state['num_observations'],
                      state['decoder_emb_sz'], state['decoder_hid_sz'], state['num_words'],
                      start_token=state['start_token'])
        tourist.load_state_dict(state['parameters'])
        return tourist


def show_samples(data, tourist, dictionary, num_samples=10, cuda=True, logger=None):
    length = len(data[0])
    observations = []
    actions = []
    messages = []

    for _ in range(num_samples):
        index = random.randint(0, length-1)
        observations.append(data[0][index])
        actions.append(data[1][index])
        messages.append(data[2][index])

    batch = create_batch(observations, actions, messages, cuda=cuda)
    obs_batch, obs_seq_len, act_batch, act_seq_len, message_batch, mask_batch = batch

    out = tourist.forward(obs_batch, obs_seq_len, act_batch, act_seq_len,
                          gt_messages=message_batch, gt_mask=mask_batch,
                          sample=True)

    preds = out['preds'].cpu().data
    logger_fn = print
    if logger:
        logger_fn = logger
    for i in range(num_samples):
        o = ''
        for obs in observations[i]:
            o += '(' + ','.join([landmark_map.i2landmark[o_ind-1] for o_ind in obs]) + ') ,'

        logger_fn('Observations: ' + o)
        logger_fn('GT: ' + dict.decode(messages[i][1:]))
        logger_fn('Sample: ' + dict.decode(preds[i, :]))
        logger_fn('-'*80)

def eval_epoch(data, tourist, batch_sz=32, cuda=True, opt=None):
    observations = data[0]
    actions = data[1]
    messages = data[2]


    total_loss, total_examples = 0.0, 0.0
    for jj in range(0, len(data[0]), batch_sz):
        batch = create_batch(observations[jj:jj + batch_sz],
                             actions[jj:jj + batch_sz],
                             messages[jj:jj + batch_sz],
                             cuda=cuda)
        obs_batch, obs_seq_len, act_batch, act_seq_len, message_batch, mask_batch = batch

        out = tourist.forward(obs_batch, obs_seq_len, act_batch, act_seq_len,
                               gt_messages=message_batch, gt_mask=mask_batch,
                               sample=False)
        loss = out['loss']
        total_loss += float(loss.data)
        total_examples += obs_batch.size(0)

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
    return total_loss/total_examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--act-emb-sz', type=int, default=32)
    parser.add_argument('--act-hid-sz', type=int, default=128)
    parser.add_argument('--obs-emb-sz', type=int, default=32)
    parser.add_argument('--obs-hid-sz', type=int, default=256)
    parser.add_argument('--decoder-emb-sz', type=int, default=128)
    parser.add_argument('--decoder-hid-sz', type=int, default=512)
    parser.add_argument('--batch-sz', type=int, default=128)

    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--exp-name', type=str, default='tourist_sl')

    args = parser.parse_args()

    exp_dir = os.path.join(os.environ['TALKTHEWALK_EXPDIR'], args.exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    logger = create_logger(os.path.join(exp_dir, 'log.txt'))
    logger.info(args)

    data_dir = os.environ.get('TALKTHEWALK_DATADIR', './data')

    train_configs = json.load(open(os.path.join(data_dir, 'talkthewalk.train.json')))
    valid_configs = json.load(open(os.path.join(data_dir, 'talkthewalk.valid.json')))
    test_configs = json.load(open(os.path.join(data_dir, 'talkthewalk.test.json')))

    dict = Dictionary(file=os.path.join(data_dir, 'dict.txt'), min_freq=3)
    landmark_map = Landmarks(neighborhoods, include_empty_corners=True)
    loader = GoldstandardFeatures(landmark_map)

    train_data = load_data(train_configs, dict, loader, landmark_map)
    valid_data = load_data(valid_configs, dict, loader, landmark_map)
    test_data = load_data(test_configs, dict, loader, landmark_map)


    tourist = Tourist(args.act_emb_sz, args.act_hid_sz, 6, args.obs_emb_sz, args.obs_hid_sz, len(landmark_map.i2landmark)+1,
                      args.decoder_emb_sz, args.decoder_hid_sz, len(dict), start_token=dict.tok2i[START_TOKEN])

    opt = optim.Adam(tourist.parameters())

    if args.cuda:
        tourist = tourist.cuda()

    best_val = 1e10

    for epoch in range(1, args.num_epochs):
        train_loss = eval_epoch(train_data, tourist, batch_sz=args.batch_sz, cuda=args.cuda, opt=opt)
        valid_loss = eval_epoch(valid_data, tourist, batch_sz=args.batch_sz, cuda=args.cuda)

        logger.info('Train loss: {}, Valid_loss: {}'.format(train_loss, valid_loss))
        show_samples(test_data, tourist, dict, cuda=args.cuda, num_samples=5, logger=logger.info)

        if valid_loss < best_val:
            best_val = valid_loss
            tourist.save(os.path.join(exp_dir, 'tourist.pt'))






