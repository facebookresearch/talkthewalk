# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from ttw.models.modules import MASC, NoMASC, CBoW

class TouristDiscrete(nn.Module):
    def __init__(self, vocab_sz, num_observations, num_actions, T=2, apply_masc=False):
        super(TouristDiscrete, self).__init__()
        self.T = T
        self.apply_masc = apply_masc
        self.vocab_sz = vocab_sz
        self.num_observations = num_observations
        self.num_actions = num_actions

        self.goldstandard_emb = CBoW(num_observations, vocab_sz, init_std=0.1)

        self.num_embeddings = T + 1
        self.obs_write_gate = nn.ParameterList()
        for _ in range(T + 1):
            self.obs_write_gate.append(nn.Parameter(torch.FloatTensor(1, vocab_sz).normal_(0.0, 0.1)))

        if self.apply_masc:
            self.action_emb = nn.Embedding(num_actions, vocab_sz)
            self.num_embeddings += T
            self.act_write_gate = nn.ParameterList()
            for _ in range(T):
                self.act_write_gate.append(nn.Parameter(torch.FloatTensor(1, vocab_sz).normal_(0.0, 0.1)))

        self.value_pred = nn.Linear((1 + int(self.apply_masc)) * self.vocab_sz, 1)

    def forward(self, batch, greedy=False):
        batch_size = batch['actions'].size(0)
        feat_emb = list()

        max_steps = self.T + 1
        for step in range(max_steps):
            emb = self.goldstandard_emb.forward(batch['goldstandard'][:, step, :])
            emb = emb * F.sigmoid(self.obs_write_gate[step])
            feat_emb.append(emb)

        act_emb = list()
        if self.apply_masc:
            for step in range(self.T):
                emb = self.action_emb.forward(batch['actions'][:, step])
                emb = emb * F.sigmoid(self.act_write_gate[step])
                act_emb.append(emb)

        out = {}
        out['comms'] = list()
        out['probs'] = list()

        feat_embeddings = sum(feat_emb)
        feat_logits = feat_embeddings
        feat_prob = F.sigmoid(feat_logits).cpu()
        feat_msg = feat_prob.bernoulli().detach()

        out['probs'].append(feat_prob)
        out['comms'].append(feat_msg)

        if self.apply_masc:
            act_embeddings = sum(act_emb)
            act_logits = act_embeddings
            act_prob = F.sigmoid(act_logits).cpu()
            act_msg = act_prob.bernoulli().detach()

            out['probs'].append(act_prob)
            out['comms'].append(act_msg)

        if self.apply_masc:
            embeddings = torch.cat([feat_embeddings, act_embeddings], 1).resize(batch_size, 2 * self.vocab_sz)
        else:
            embeddings = feat_embeddings
        out['baseline'] = self.value_pred(embeddings)

        return out

    def save(self, path):
        state = dict()
        state['vocab_sz'] = self.vocab_sz
        state['num_observations'] = self.num_observations
        state['num_actions'] = self.num_actions
        state['T'] = self.T
        state['apply_masc'] = self.apply_masc
        state['parameters'] = self.state_dict()
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        tourist = cls(state['vocab_sz'], state['num_observations'], state['num_actions'],
                      T=state['T'], apply_masc=state['apply_masc'])
        tourist.load_state_dict(state['parameters'])
        return tourist


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

        self.loss = nn.CrossEntropyLoss(reduce=False)


    def forward(self, message, batch):
        msg_obs = self.obs_emb_fn(message[0])
        batch_size = message[0].size(0)

        landmark_emb = self.emb_map.forward(batch['landmarks']).permute(0, 3, 1, 2)
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

        out = dict()
        logits = torch.bmm(landmarks, msg_obs.unsqueeze(-1)).squeeze(-1)
        out['prob'] = F.softmax(logits, 1)
        y_true = (batch['target'][:, 0] * 4 + batch['target'][:, 1])

        out['loss'] = self.loss(logits, y_true)
        out['acc'] = sum(
            [1.0 for pred, target in zip(out['prob'].max(1)[1].data.cpu().numpy(), y_true.data.cpu().numpy()) if
             pred == target]) / y_true.size(0)
        return out

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
