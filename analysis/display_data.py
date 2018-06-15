# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Show all samples from the train set."""
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='./data', help='Path to talkthewalk dataset')

args = parser.parse_args()

f = os.path.join(args.data_dir, 'talkthewalk.train.json')

f = './data/talkthewalk.train.json'
data = json.load(open(f))

for sample in data:
    tourist_action_list = list()
    for turn in sample['dialog']:
        if turn['id'] == 'Tourist' and 'ACTION' in turn['text']:
            tourist_action_list.append(turn['text'])
        else:
            if len(tourist_action_list) > 0:
                print("Tourist:  " + ' '.join(tourist_action_list))
                tourist_action_list = list()
            if turn['id'] == 'Tourist':
                print('Tourist:  ' + turn['text'])
            else:
                print('Guide:    ' + turn['text'])

    print('-'*80)
