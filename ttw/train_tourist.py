import os
import json
import argparse
import random
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

from ttw.data_loader import TalkTheWalkLanguage, get_collate_fn
from ttw.modules import CBoW
from ttw.dict import START_TOKEN, END_TOKEN
from ttw.utils import create_logger
from ttw.beam_search import SequenceGenerator


class TouristLanguage(nn.Module):

    def __init__(self, act_emb_sz, act_hid_sz, num_actions, obs_emb_sz, obs_hid_sz, num_observations,
                 decoder_emb_sz, decoder_hid_sz, num_words, start_token=1, end_token=2):
        super(TouristLanguage, self).__init__()
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
        self.end_token = end_token

    def pick_hidden_state(self, states, seq_lens):
        batch_size = seq_lens.size(0)
        return states[torch.arange(batch_size).long(), seq_lens - 1, :]

    def encode(self, observations, obs_seq_len, actions, act_seq_len):
        obs_inp_emb = self.obs_emb_fn(observations)
        obs_h, _ = self.obs_encoder(obs_inp_emb)
        observation_emb = self.pick_hidden_state(obs_h, obs_seq_len)

        act_inp_emb = self.act_emb_fn(actions)
        action_embs, _ = self.act_encoder(act_inp_emb)
        action_emb = self.pick_hidden_state(action_embs, act_seq_len)

        context_emb = torch.cat([observation_emb, action_emb], 1)
        context_emb = self.context_linear.forward(context_emb)

        return context_emb

    def forward(self, batch,
                decoding_strategy='beam_search', max_sample_length=20, train=True):
        batch_size = batch['observations'].size(0)
        obs_seq_len = batch['observations_mask'][:, :, 0].sum(1).long()
        act_seq_len = batch['actions_mask'].sum(1).long()
        context_emb = self.encode(batch['observations'], obs_seq_len, batch['actions'], act_seq_len)

        if train:
            # teacher forcing
            assert('utterance_mask' in batch.keys() and 'utterance' in batch.keys())
            inp = batch['utterance'][:, :-1]
            tgt = batch['utterance'][:, 1:]

            inp_emb = self.emb_fn.forward(inp)

            # concatenate external emb
            context_emb = context_emb.view(batch_size, 1, self.decoder_emb_sz).repeat(1, inp_emb.size(1), 1)
            inp_emb = torch.cat([inp_emb, context_emb], 2)

            hs, _ = self.decoder(inp_emb)

            score = self.out_linear(hs)

            loss = 0.0
            mask = batch['utterance_mask'][:, 1:]

            for j in range(score.size(1)):
                flat_mask = mask[:, j]
                flat_score = score[:, j, :]
                flat_tgt = tgt[:, j]
                nll = self.loss(flat_score, flat_tgt)
                loss += (flat_mask*nll).sum()

            out = {}
            out['loss'] = loss
        else:
            if decoding_strategy in ['greedy', 'sample']:
                preds = []
                probs = []

                input_ind = torch.LongTensor([self.start_token] * batch_size)
                hs = Variable(torch.FloatTensor(1, batch_size, self.decoder_hid_sz).fill_(0.0))
                mask = Variable(torch.FloatTensor(batch_size, max_sample_length).zero_())
                eos = torch.ByteTensor([0]*batch_size)
                if batch['observations'].is_cuda:
                    hs = hs.cuda()
                    eos = eos.cuda()
                    mask = mask.cuda()
                    input_ind = input_ind.cuda()

                for k in range(max_sample_length):
                    inp_emb = self.emb_fn.forward(input_ind.unsqueeze(-1))

                    context_emb = context_emb.view(batch_size, 1, self.decoder_emb_sz).repeat(1, inp_emb.size(1), 1)
                    inp_emb = torch.cat([inp_emb, context_emb], 2)

                    _, hs = self.decoder(inp_emb, hs)

                    prob = F.softmax(self.out_linear(hs.squeeze(0)), dim=-1)
                    if decoding_strategy == 'greedy':
                        _, samples = prob.max(1)
                        samples = samples.unsqueeze(-1)
                    else:
                        samples = prob.multinomial(1)
                    mask[:, k] = 1.0 - eos.float()

                    eos = eos | (samples == self.end_token).squeeze()

                    preds.append(samples)
                    probs.append(prob.unsqueeze(1))
                    input_ind = samples.squeeze()

                out = {}
                out['preds'] = torch.cat(preds, 1)
                out['mask'] = mask
                out['probs'] = torch.cat(probs, 1)
            elif decoding_strategy == 'beam_search':
                def _step_fn(input, hidden, context, k=4):
                    input = Variable(torch.LongTensor(input)).squeeze().cuda()
                    hidden = Variable(torch.FloatTensor(hidden)).unsqueeze(0).cuda()
                    context = Variable(torch.FloatTensor(context)).unsqueeze(1).cuda()
                    prob, hs = self.step(input, hidden, context)
                    logprobs = torch.log(prob)
                    logprobs, words = logprobs.topk(k, 1)
                    hs = hs.squeeze().cpu().data.numpy()

                    return words, logprobs, hs

                seq_gen = SequenceGenerator(_step_fn, self.end_token, max_sequence_length=max_sample_length, beam_size=10, length_normalization_factor=0.8)
                start_tokens = [[self.start_token] for _ in range(batch_size)]
                hidden = [[0.0]*self.decoder_hid_sz]*batch_size
                beam_out = seq_gen.beam_search(start_tokens, hidden, context_emb.cpu().data.numpy())
                pred_tensor = torch.LongTensor(batch_size, max_sample_length).zero_()
                mask_tensor = torch.FloatTensor(batch_size, max_sample_length).zero_()

                for i, seq in enumerate(beam_out):
                    pred_tensor[i, :len(seq.output)] = torch.LongTensor(seq.output[1:])
                    mask_tensor[i, :(len(seq.output)-1)] = 1.0

                out = {}
                out['preds'] = Variable(pred_tensor)
                out['mask'] = Variable(mask_tensor)

                if batch['observations'].is_cuda:
                    out['preds'] = out['preds'].cuda()
                    out['mask'] = out['mask'].cuda()

        return out


    def step(self, input_ind, hs, context_emb):
        inp_emb = self.emb_fn.forward(input_ind.unsqueeze(-1))
        inp_emb = torch.cat([inp_emb, context_emb], 2)

        _, hs = self.decoder(inp_emb, hs)

        prob = F.softmax(self.out_linear(hs.squeeze(0)), dim=-1)
        return prob, hs


    def save(self, path):
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


def show_samples(dataset, tourist, num_samples=10, cuda=True, logger=None, decoding_strategy='sample', indices=None):
    if indices is None:
        indices = list()
        for _ in range(num_samples):
            indices.append(random.randint(0, len(dataset)-1))

    collate_fn = get_collate_fn(cuda)

    data = [dataset[ind] for ind in indices]
    batch = collate_fn(data)

    out = tourist.forward(batch,
                          decoding_strategy=decoding_strategy, train=False)

    preds = out['preds'].cpu().data
    logger_fn = print
    if logger:
        logger_fn = logger

    for i in range(len(indices)):
        o = ''
        for obs in data[i]['observations']:
            o += '(' + ','.join([dataset.map.landmark_dict.decode(o_ind) for o_ind in obs]) + ') ,'
        # a = ', '.join([i2act[a_ind] for a_ind in actions[i]])
        a = ','.join([dataset.act_dict.decode_agnostic(a_ind) for a_ind in data[i]['actions']])

        logger_fn('Observations: ' + o)
        logger_fn('Actions: ' + a)
        logger_fn('GT: ' + dataset.dict.decode(batch['utterance'][i, 1:]))
        logger_fn('Sample: ' + dataset.dict.decode(preds[i, :]))
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
    parser.add_argument('--act-emb-sz', type=int, default=32)
    parser.add_argument('--act-hid-sz', type=int, default=128)
    parser.add_argument('--obs-emb-sz', type=int, default=32)
    parser.add_argument('--obs-hid-sz', type=int, default=128)
    parser.add_argument('--decoder-emb-sz', type=int, default=128)
    parser.add_argument('--decoder-hid-sz', type=int, default=512)
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






