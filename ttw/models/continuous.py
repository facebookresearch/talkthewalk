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

class TouristContinuous(nn.Module):

    def __init__(self, vocab_sz, num_observations, num_actions, T=2, apply_masc=True):
        super(TouristContinuous, self).__init__()
        self.vocab_sz = vocab_sz
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.apply_masc = apply_masc
        self.T = T
        self.goldstandard_emb = CBoW(num_observations, vocab_sz, init_std=0.1, padding_idx=0)

        self.obs_write_gate = nn.ParameterList()
        for _ in range(T + 1):
            self.obs_write_gate.append(nn.Parameter(torch.FloatTensor(1, vocab_sz).normal_(0.0, 0.1)))

        if apply_masc:
            self.action_emb = nn.Embedding(num_actions, vocab_sz, padding_idx=0)
            self.act_write_gate = nn.Parameter(torch.FloatTensor(1, T, vocab_sz).normal_(0.0, 0.1))


    def forward(self, batch):
        out = dict()
        embs = list()
        for step in range(self.T + 1):
            emb = self.goldstandard_emb.forward(batch['goldstandard'][:, step, :])
            emb = emb * F.sigmoid(self.obs_write_gate[step])
            embs.append(emb)
        out['obs'] = sum(embs)

        out['act'] = None
        if self.apply_masc:
            action_emb = self.action_emb.forward(batch['actions'])
            action_emb *= F.sigmoid(self.act_write_gate)
            out['act'] = action_emb.sum(dim=1)

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


class GuideContinuous(nn.Module):

    def __init__(self, in_vocab_sz, num_landmarks, T=2, apply_masc=True):
        super(GuideContinuous, self).__init__()
        self.in_vocab_sz = in_vocab_sz
        self.num_landmarks = num_landmarks
        self.apply_masc = apply_masc
        self.T = T

        self.landmark_write_gate = nn.ParameterList()
        for _ in range(self.T + 1):
            self.landmark_write_gate.append(nn.Parameter(torch.FloatTensor(1, in_vocab_sz, 1, 1).normal_(0.0, 0.1)))

        self.cbow_fn = CBoW(num_landmarks, in_vocab_sz, init_std=0.01)

        if self.apply_masc:
            self.masc_fn = MASC(in_vocab_sz)
            self.extract_fns = nn.ModuleList()
            for _ in range(T):
                self.extract_fns.append(nn.Linear(in_vocab_sz, 9))
        else:
            self.masc_fn = NoMASC(in_vocab_sz)

        self.loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, msg, batch):
        obs_msg, act_msg = msg['obs'], msg['act']

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

        out = dict()
        logits = torch.bmm(landmarks, obs_msg.unsqueeze(-1)).squeeze(-1)
        out['prob'] = F.softmax(logits, dim=1)


        y_true = (batch['target'][:, 0]*4 + batch['target'][:, 1])

        out['loss'] = self.loss(logits, y_true)
        out['acc'] = sum([1.0 for pred, target in zip(out['prob'].max(1)[1].data.cpu().numpy(), y_true.data.cpu().numpy()) if pred == target])/y_true.size(0)
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
