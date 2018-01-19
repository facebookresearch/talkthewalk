import torch
import torch.nn as nn
import torch.nn.functional as F

class MapEmbedding(nn.Module):

    def __init__(self, num_tokens, emb_size):
        super(MapEmbedding, self).__init__()
        self.emb_landmark = nn.Embedding(num_tokens, emb_size, padding_idx=0)
        self.emb_size = emb_size

    def forward(self, x):
        bsz = x.size(0)
        out = []
        for i in range(bsz):
            out.append(self.emb_landmark.forward(x[i, :, :]).sum(1).unsqueeze(0))
        return torch.cat(out, 0)

class Guide(nn.Module):
    def __init__(self, in_vocab_sz, num_landmarks, embed_sz=20):
        super(Guide, self).__init__()
        self.in_vocab_sz = in_vocab_sz
        self.num_landmarks = num_landmarks
        self.embed_sz = embed_sz
        self.emb_map = MapEmbedding(num_landmarks, embed_sz)
        self.emb_comms = nn.Linear(in_vocab_sz, embed_sz)

    def forward(self, message, landmarks, mask):
        msg = self.emb_comms(message)
        lmarks = self.emb_map.forward(landmarks)

        msg = F.relu(msg)
        lmarks = F.relu(lmarks)
        # msg = F.dropout(msg, p=0.2, training=self.training)
        # lmarks = F.dropout(lmarks, p=0.2, training=self.training)

        bsz = message.size(0)
        logits = []
        for i in range(bsz):
            score = torch.matmul(lmarks[i, :, :], msg[i, :])
            score += (1.0 - mask[i, :])*-1.e36
            logits.append(score.unsqueeze(0))

        return F.softmax(torch.cat(logits), dim=1)

    def save(self, path):
        state = dict()
        state['in_vocab_sz'] = self.in_vocab_sz
        state['num_landmarks'] = self.num_landmarks
        state['embed_sz'] = self.embed_sz
        state['parameters'] = self.state_dict()
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        guide = cls(state['in_vocab_sz'], state['num_landmarks'], embed_sz=state['embed_sz'])
        guide.load_state_dict(state['parameters'])
        return guide


class Tourist(nn.Module):
    def __init__(self, in_vocab_sz, out_vocab_sz, embed_sz=20):
        super(Tourist, self).__init__()
        self.in_vocab_sz = in_vocab_sz
        self.out_vocab_sz = out_vocab_sz
        self.embed_sz = embed_sz
        self.emb_obsv = nn.Embedding(in_vocab_sz, embed_sz, padding_idx=0)
        self.out_comms = nn.Linear(embed_sz, out_vocab_sz)
        self.value_pred = nn.Linear(embed_sz, 1)
        self.in_vocab_sz = in_vocab_sz

    def forward(self, observation, greedy=False):
        hid = self.emb_obsv(observation)
        hid = hid.sum(0)
        hid = F.relu(hid)
        # hid = F.dropout(hid, p=0.2, training=self.training)

        probs = F.sigmoid(self.out_comms(hid))
        comms = probs.cpu().bernoulli()
        if greedy:
            comms = torch.FloatTensor(self.in_vocab_sz).zero_()
            comms[probs>0.5] = 1.0

        value = self.value_pred(hid.detach())
        return comms, probs, value

    def save(self, path):
        state = dict()
        state['in_vocab_sz'] = self.in_vocab_sz
        state['out_vocab_sz'] = self.out_vocab_sz
        state['embed_sz'] = self.embed_sz
        state['parameters'] = self.state_dict()
        torch.save(state, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        tourist = cls(state['in_vocab_sz'], state['out_vocab_sz'], embed_sz=state['embed_sz'])
        tourist.load_state_dict(state['parameters'])
        return tourist