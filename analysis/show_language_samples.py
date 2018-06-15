# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random
import argparse


from ttw.data_loader import TalkTheWalkLanguage
from ttw.models import TouristLanguage

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='./data', help='Path to talkthewalk dataset')
parser.add_argument('--tourist-model', type=str, default='Path to tourist checkpoint')

args = parser.parse_args()

train_data = TalkTheWalkLanguage(args.data_dir, 'train')

tourist_sl = TouristLanguage.load(args.tourist_model).cuda()

indices = []
for _ in range(5):
    indices.append(random.randint(0, len(train_data) - 1))

indices = [32, 245, 560, 750, 1200, 2467, 2500]
print('supervised, greedy')
tourist_sl.show_samples(train_data, indices=indices, decoding_strategy='greedy')
print(); print()
print('supervised, beam')
tourist_sl.show_samples(train_data, indices=indices, beam_width=8, decoding_strategy='beam_search')
