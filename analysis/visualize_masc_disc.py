import numpy
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torch.autograd import Variable
from talkthewalk.predict_location_discrete import Tourist, Guide

tourist = Tourist.load('/u/devries/Documents/talkthewalk/results/disc_masc_3/tourist.pt').cuda()
guide = Guide.load('/u/devries/Documents/talkthewalk/results/disc_masc_3/guide.pt').cuda()

actions = ['up', 'right', 'down', 'left']

examples = [[0, 1, 2], [1, 3, 0], [2, 3, 0], [3, 0, 1]]
for i in range(len(examples)):
    X_batch = {'goldstandard': Variable(torch.LongTensor(1, 3, 1).fill_(0)).cuda()}
    action_batch = Variable(torch.LongTensor([examples[i]])).cuda()

    t_comms, t_probs, t_val = tourist(X_batch, action_batch)
    for j in range(len(examples[0])):
        out = guide.action_emb[j](t_comms[1].cuda())
        mask = F.softmax(out).view(1, 1, 3, 3).squeeze().cpu().data.numpy()

        mask = mask.transpose()
        mask = numpy.flip(mask, 0)
        # mask = numpy.flip(mask, 1)

        plt.matshow(mask)
        plt.savefig('{}_{}.png'.format(i, j))

