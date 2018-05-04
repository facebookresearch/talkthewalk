import torch
import torch.nn as nn
import torch.nn.functional as F
import operator

from functools import reduce
from torch.autograd import Variable

class CBoW(nn.Module):

    def __init__(self, num_tokens, emb_size, init_std=1, padding_idx=None):
        super(CBoW, self).__init__()
        self.emb_fn = nn.Embedding(num_tokens, emb_size, padding_idx=padding_idx)
        if init_std != 1.0:
            self.emb_fn.weight.data.normal_(0.0, init_std)
        self.emb_size = emb_size

    def forward(self, x):
        in_shape = x.size()
        num_elem = reduce(operator.mul, in_shape)
        flat_x = x.view(num_elem)
        flat_emb = self.emb_fn.forward(flat_x)
        emb = flat_emb.view(*(in_shape+(self.emb_size,)))
        return emb.sum(dim=-2)


class MASC(nn.Module):

    def __init__(self, hidden_sz, apply_masc=True):
        super(MASC, self).__init__()
        self.conv_weight = nn.Parameter(torch.FloatTensor(
            hidden_sz, hidden_sz, 3, 3))
        std = 1.0 / (hidden_sz * 9)
        self.conv_weight.data.uniform_(-std, std)
        self.apply_masc = apply_masc

    def forward(self, inp, action_out):
        batch_size = inp.size(0)
        out = inp.clone().zero_()

        for i in range(batch_size):
            selected_inp = inp[i, :, :, :].unsqueeze(0)
            mask = F.softmax(action_out[i], dim=0).view(1, 1, 3, 3)
            weight = mask * self.conv_weight
            out[i, :, :, :] = F.conv2d(selected_inp, weight, padding=1).squeeze(0)
        return out

    def forward_no_masc(self, input):
        assert not self.apply_masc, "`apply_masc` needs to be set to False before you can apply this fn"

        mask = torch.FloatTensor(1, 1, 3, 3).zero_()
        mask[0, 0, 0, 1] = 1.0
        mask[0, 0, 1, 0] = 1.0
        mask[0, 0, 2, 1] = 1.0
        mask[0, 0, 1, 2] = 1.0

        mask = Variable(mask)
        if input.is_cuda:
            mask.cuda()

        weight = self.conv_weight * mask
        return F.conv2d(input, weight, padding=1)


class ControlStep(nn.Module):

    def __init__(self, hidden_sz):
        super(ControlStep, self).__init__()
        self.control_updater = nn.Linear(2 * hidden_sz, hidden_sz)
        self.hop_fn = AttentionHop()

    def forward(self, inp_seq, mask, query):
        extracted_msg = self.hop_fn.forward(inp_seq, mask, query)
        conc_emb = torch.cat([query, extracted_msg], 1)
        control_emb = self.control_updater.forward(conc_emb)
        return extracted_msg, control_emb


class AttentionHop(nn.Module):

    def forward(self, inp_seq, mask, query):
        score = torch.bmm(inp_seq, query.unsqueeze(-1)).squeeze(-1)
        score = score - 1e30 * (1.0 - mask)
        att_score = F.softmax(score, dim=-1)
        extracted_msg = torch.bmm(att_score.unsqueeze(1), inp_seq).squeeze()
        return extracted_msg



