# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import copy
import operator
import os
import json
import random
import itertools

from ttw.data_loader import Map, GoldstandardFeatures
from ttw.env import step_agnostic, step_aware

def init_paths_agnostic(neighborhood, boundaries, goldstandard_features):
    paths = list()
    for i in range(4):
        for j in range(4):
            path = dict()
            path['loc'] = [boundaries[0] + i, boundaries[1] + j, 0]
            path['seq_of_landmarks'] = [goldstandard_features.get(neighborhood, path['loc'])]
            paths.append(path)
    return paths

def init_paths_aware(neighborhood, boundaries, goldstandard_features):
    paths = list()
    for i in range(4):
        for j in range(4):
            for k in range(4):
                path = dict()
                path['loc'] = [boundaries[0] + i, boundaries[1] + j, k]
                path['seq_of_landmarks'] = [goldstandard_features.get(neighborhood, path['loc'])]
                paths.append(path)
    return paths

def prediction_upperbound(seq_of_landmarks, goldstandard_features, neighborhood, boundaries, loc, actions=None, step_fn=step_agnostic, action_space=[]):
    depth = len(seq_of_landmarks)

    if actions is not None:
        assert len(actions) == depth - 1

    if len(action_space) == 4:
        paths = init_paths_agnostic(neighborhood, boundaries, goldstandard_features)
    else:
        paths = init_paths_aware(neighborhood, boundaries, goldstandard_features)

    for i in range(4):
        for j in range(4):
            path = dict()
            path['loc'] = [boundaries[0] + i, boundaries[1] + j, 0]
            path['seq_of_landmarks'] = [goldstandard_features.get(neighborhood, path['loc'])]
            paths.append(path)

    for d in range(depth-1):
        new_paths = list()
        for path in paths:
            for act in action_space:
                if actions is None or actions[d] == act:
                    path_new = copy.deepcopy(path)
                    path_new['loc'] = step_fn(act, path['loc'], boundaries)
                    path_new['seq_of_landmarks'].append(goldstandard_features.get(neighborhood, path_new['loc']))
                    new_paths.append(path_new)
        paths = new_paths

    loc2cnt = dict()
    for path in paths:
        if all([l1 == l2 for l1, l2 in zip(path['seq_of_landmarks'], seq_of_landmarks)]):
            if (path['loc'][0], path['loc'][1]) in loc2cnt:
                loc2cnt[(path['loc'][0], path['loc'][1])] += 1
            else:
                loc2cnt[(path['loc'][0], path['loc'][1])] = 1

    # find maximum
    selected_loc = max(loc2cnt.items(), key=operator.itemgetter(1))[0]
    acc = float(selected_loc[0] == loc[0] and selected_loc[1] == loc[1])
    return acc


def process(configs, feature_loaders, num_steps, step_fn=step_agnostic, action_space=['UP', 'LEFT', 'RIGHT', 'DOWN'], condition_on_action=False):
    correct, cnt = 0, 0

    all_possible_actions = [[]]
    if num_steps > 1:
        action_set = [action_space] * (num_steps - 1)
        all_possible_actions = list(itertools.product(*action_set))

    for config in configs:
        for a in all_possible_actions:
            neighborhood = config['neighborhood']
            target_loc = config['target_location']
            boundaries = config['boundaries']

            obs = {k: list() for k in feature_loaders.keys()}
            actions = list()
            loc = copy.deepcopy(config['target_location'])
            loc[2] = random.randint(0, 3)
            for p in range(num_steps):
                for k, feature_loader in feature_loaders.items():
                    obs[k].append(feature_loader.get(neighborhood, loc))

                if p != num_steps - 1:
                    sampled_act = a[p]
                    actions.append(sampled_act)
                    loc = step_fn(sampled_act, loc, boundaries)
            act_seq = None
            if condition_on_action:
                act_seq = a
            correct += prediction_upperbound(obs['goldstandard'], feature_loaders['goldstandard'],
                                             neighborhood, boundaries, loc,
                                             step_fn=step_fn, actions=act_seq, action_space=action_space)
            cnt += 1

    return correct/cnt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to talkthewalk dataset')
    parser.add_argument('--orientation-aware', action='store_true',
                        help='If true, take into account orientation of tourist')
    parser.add_argument('--max-T', type=int, default=3,
                        help='Maximum length of trajectory to calculate upperbound for')
    parser.add_argument('--condition-on-action',  action='store_true',
                        help='If true, only consider paths constructed from a specific sequence of actions')

    args = parser.parse_args()

    train_configs = json.load(open(os.path.join(args.data_dir, 'configurations.train.json')))
    valid_configs = json.load(open(os.path.join(args.data_dir, 'configurations.valid.json')))
    test_configs = json.load(open(os.path.join(args.data_dir, 'configurations.test.json')))

    neighborhoods = ['fidi', 'hellskitchen', 'williamsburg', 'uppereast', 'eastvillage']
    landmark_map = Map(args.data_dir, neighborhoods, include_empty_corners=True)

    if args.orientation_aware:
        step_fn = step_aware
        action_space = ['ACTION:FORWARD', 'ACTION:TURNLEFT', 'ACTION:TURNRIGHT']
    else:
        step_fn = step_agnostic
        action_space = ['UP', 'LEFT', 'RIGHT', 'DOWN']

    feature_loaders = dict()
    feature_loaders['goldstandard'] = GoldstandardFeatures(landmark_map, orientation_aware=args.orientation_aware)

    for T in range(0, args.max_T+1):
        train_upp = process(train_configs, feature_loaders, T+1, step_fn=step_fn,
                            action_space=action_space, condition_on_action=args.condition_on_action)
        valid_upp = process(valid_configs, feature_loaders, T+1, step_fn=step_fn,
                            action_space=action_space, condition_on_action=args.condition_on_action)
        test_upp = process(test_configs, feature_loaders, T+1, step_fn=step_fn,
                           action_space=action_space, condition_on_action=args.condition_on_action)

        print("T=%i, %.2f, %.2f, %.2f" % (T, train_upp*100, valid_upp*100, test_upp*100))
