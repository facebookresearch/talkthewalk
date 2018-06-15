# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import numpy
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torch.autograd import Variable
from ttw.models.discrete import TouristDiscrete, GuideDiscrete
from ttw.data_loader import ActionAgnosticDictionary

plt.switch_backend('agg')
plt.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument('--tourist-model', type=str, help='Path to tourist checkpoint')
parser.add_argument('--guide-model', type=str, help='Path to guide checkpoint')

args = parser.parse_args()

tourist = TouristDiscrete.load(args.tourist_model).cuda()
guide = GuideDiscrete.load(args.guide_model).cuda()

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
act_dict = ActionAgnosticDictionary()

examples = [[0, 1, 2], [1, 3, 0], [2, 3, 0], [3, 0, 1]]
for i in range(len(examples)):
    batch = dict()
    batch['goldstandard'] = Variable(torch.LongTensor(1, 4, 1).fill_(0)).cuda()
    batch['actions'] = Variable(torch.LongTensor([[act_dict.encode(actions[k]) for k in examples[i]]])).cuda()

    t_out = tourist(batch)
    for j in range(len(examples[0])):
        out = guide.action_emb[j](t_out['comms'][1].cuda())
        mask = F.softmax(out).view(1, 1, 3, 3).squeeze().cpu().data.numpy()

        mask = mask.transpose()
        mask = numpy.flip(mask, 0)
        # mask = numpy.flip(mask, 1)

        plt.matshow(mask)
        plt.savefig('{}_{}.png'.format(i, j))
