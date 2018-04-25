import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from torch.autograd import Variable
from dict import START_TOKEN, END_TOKEN, UNK_TOKEN, PAD_TOKEN


class KVMemnn(nn.Module):
    """docstring for [object Object]."""
    def __init__(self, args, dictionary):
        super().__init__()
        num_words = len(dictionary)
        self.dictionary = dictionary
        self.trg_pad_idx = dictionary.tok2i[PAD_TOKEN]
        self.emb_dim = args.embedding
        self.lt = nn.Embedding(num_words, args.embedding_dim, self.trg_pad_idx)
        self.encoder = Encoder(self.lt, dictionary)
        if args.share_embeddings:
            self.encoder2 = self.encoder
        else:
            self.lt2 = nn.Embedding(num_words, args.embedding_dim, self.trg_pad_idx)
            self.encoder2 = Encoder(self.lt2, dictionary)
        self.args = args
        self.softmax = nn.Softmax(dim=1)
        self.cosine = nn.CosineSimilarity()
        self.linear1 = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.linear2 = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.hops = 1
        self.lins = 0

    def forward(self, xs, mems, ys=None, cands=None):
        xs_enc = []
        xs_emb = self.encoder(xs)

        if len(mems) > 0 and self.hops > 0:
            mem_enc = []
            for m in mems:
                mem_enc.append(self.encoder(m))
            mem_enc.append(xs_emb)
            mems_enc = torch.cat(mem_enc)
            layer2 = self.cosine(xs_emb, mems_enc).unsqueeze(0)
            layer3 = self.softmax(layer2)
            lhs_emb = torch.mm(layer3, mems_enc)

            if self.lins > 0:
                lhs_emb = self.lin1(lhs_emb)
            if self.hops > 1:
                layer4 = self.cosine(lhs_emb, mems_enc).unsqueeze(0)
                layer5 = self.softmax(layer4)
                lhs_emb = torch.mm(layer5, mems_enc)
                if self.lins > 1:
                    lhs_emb = self.lin2(lhs_emb)
        else:
            if self.lins > 0:
                lhs_emb = self.lin1(xs_emb)
            else:
                lhs_emb = xs_emb
        if ys is not None:
            # training
            ys_enc = []
            xs_enc.append(lhs_emb)
            ys_enc.append(self.encoder2(ys))
            for c in cands:
                xs_enc.append(lhs_emb)
                c_emb = self.encoder2(c)
                ys_enc.append(c_emb)
        else:
            # test
            ys_enc = []
            for c in cands:
                xs_enc.append(lhs_emb)
                c_emb = self.encoder2(c)
                ys_enc.append(c_emb)
        return torch.cat(xs_enc), torch.cat(ys_enc)


class Encoder(nn.Module):
    def __init__(self, lt, dictionary):
        super().__init__()
        self.lt = lt
        if dictionary is not None:
            n_words = len(dictionary)
            freqs = torch.Tensor(n_words)
            for i in range(n_words):
                ind = dictionary.i2tok[i]
                freq = dictionary.tok2cnt[ind]
                freqs[i] = 1.0 / (1.0 + math.log(1.0 + freq))
            self.freqs = freqs
        else:
            self.freqs = None

    def forward(self, xs):
        xs_emb = self.lt(xs)
        if self.freqs is not None:
            # tfidf embeddings
            length = xs.size(1)
            w = Variable(torch.Tensor(length))
            for i in range(length):
                w[i] = self.freqs[xs.data[0][i]]
            w = w.mul(1 / w.norm())
            xs_emb = xs_emb.squeeze(0).t().matmul(w.unsqueeze(1)).t()
        else:
            # basic embeddings (faster)
            xs_emb = xs_emb.mean(1)
        return xs_emb
