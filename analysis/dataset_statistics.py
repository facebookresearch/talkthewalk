# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Calculate some basic statistics over the train set. """

import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='./data', help='Path to talkthewalk dataset')

args = parser.parse_args()

f = os.path.join(args.data_dir, 'talkthewalk.train.json')
data = json.load(open(f))

worker_ids = dict()

k = 0.
tourist_forward = 0.0
tourist_actions = 0.0
tourist_utterance = 0.
utt_length = 0.
guide_utterance = 0.
length = 0.

for sample in data:
    if sample['tourist_worker_id'] not in worker_ids:
        worker_ids[sample['tourist_worker_id']] = True
    if sample['guide_worker_id'] not in worker_ids:
        worker_ids[sample['guide_worker_id']] = True

    for turn in sample['dialog']:
        if turn['id'] == 'Tourist':
            if 'ACTION:' in turn['text']:
                tourist_actions += 1
                if 'FORWARD' in turn['text']:
                    tourist_forward += 1
            else:
                tourist_utterance += 1
        else:
            guide_utterance += 1
    k+=1


print("Number of Turkers", len(worker_ids))
print("Average number of turns per dialogue", (tourist_actions+tourist_utterance+guide_utterance)/k)
print("Average number of actions per dialogue", tourist_actions/k)
print("Average number of tourist utterance per dialogue", tourist_utterance/k)
print("Average number of guide utterance per dialogue", guide_utterance/k)
print("Average number of forward per dialogue", tourist_forward/k)
