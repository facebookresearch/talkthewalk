import json
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torch.autograd import Variable
from predict_location_discrete import Tourist, Guide

tourist = Tourist.load('/u/devries/emergent_a4/tourist.pt').cuda()
guide = Guide.load('/u/devries/emergent_a4/guide.pt').cuda()

actions = ['right', 'left', 'down', 'up']

for i in range(2):
    for j in range(4):
        X_batch = {'goldstandard': Variable(torch.LongTensor(1, 4, 1).fill_(0)).cuda()}
        action_batch = Variable(torch.LongTensor([[3, 0, 0]])).cuda()
        action_batch[0, i] = j

        t_comms, t_probs, t_val = tourist(X_batch, action_batch)
        out = guide.action_emb[0](t_comms[1].cuda())
        mask = F.softmax(out).reshape(3, 3).cpu().data.numpy()
        plt.matshow(mask)
        plt.savefig('{}_{}.png'.format(i, actions[j]))

