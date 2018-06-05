# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import copy
import json
import os
import random
import time

import numpy
import torch

from ttw.models import TouristContinuous, GuideContinuous, TouristDiscrete, GuideDiscrete, TouristLanguage, \
    GuideLanguage
from ttw.data_loader import Map, step_aware, get_collate_fn, GoldstandardFeatures, TalkTheWalkEmergent, ActionAgnosticDictionary
from ttw.dict import Dictionary

def evaluate(configs, predict_location_fn, collate_fn, map, feature_loader, random_walk=True, T=2,
             communication='discrete', dict=None):
    correct, total = 0.0, 0.0
    num_actions = []
    log = []
    act_dict = ActionAgnosticDictionary()

    for config in configs:
        neighborhood = config['neighborhood']
        boundaries = config['boundaries']
        target_loc = config['target_location']
        entry = {'neighborhood': neighborhood,
                 'boundaries': boundaries,
                 'target_location': target_loc,
                 'landmarks': map.get_unprocessed_landmarks(neighborhood, boundaries),
                 'dialog': []}

        landmarks, target_index = map.get_landmarks(config['neighborhood'], boundaries, target_loc)

        flat_target_index = target_index[0] * 4 + target_index[1]

        max_eval = 3
        loc = [boundaries[0] + random.randint(0, 3), boundaries[1] + random.randint(0, 3), 0]
        observations = [feature_loader.get(config['neighborhood'], loc)]
        actions = []
        locations = [loc]
        predicted = list()

        entry['start_location'] = copy.deepcopy(loc)
        t = time.time()

        for step in range(150):
            if len(actions) == T:
                batch = {}
                batch['goldstandard'] = observations
                batch['actions'] = actions
                batch['landmarks'] = landmarks
                batch['target'] = target_index

                batch = collate_fn([batch])

                prob, t_comms = predict_location_fn(batch)
                if communication == 'discrete':
                    entry['dialog'].append({'id': 'Tourist', 'episode_done': False,
                                            'text': ''.join(['%0.0f' % x for x in t_comms[0].cpu().data.numpy()[0, :]]),
                                            'time': t})
                elif communication == 'natural':
                    entry['dialog'].append({'id': 'Tourist', 'episode_done': False,
                                            'text': dict.decode(t_comms[0]),
                                            'time': t})
                    t += 1


                prob_array = [[0 for _ in range(4)] for _ in range(4)]
                prob_data = prob.squeeze().cpu().data.numpy()

                for i in range(prob_data.shape[0]):
                    prob_array[i // 4][i % 4] = float(prob_data[i])

                entry['dialog'].append({'id': 'Guide', 'episode_done': False, 'text': prob_array})
                t += 1

                sampled_index = torch.multinomial(prob, 1)
                # _, sampled_index = torch.max(prob, 1)
                if sampled_index == flat_target_index:
                    entry['dialog'].append(
                        {'id': 'Guide', 'episode_done': False, 'text': 'EVALUATE_LOCATION', 'time': t})
                    t += 1
                    predicted.append(locations[0])
                    if locations[0][0] == target_loc[0] and locations[0][1] == target_loc[1]:
                        correct += 1
                        num_actions.append(step)
                        break
                    else:
                        max_eval -= 1
                        if max_eval <= 0:
                            num_actions.append(step)
                            break

            if random_walk:
                act = ['UP', 'DOWN', 'RIGHT', 'LEFT'][random.randint(0, 3)]
                act_id = act_dict.encode(act)
                actions.append(act_id)
                act_orientation = act_dict.act_to_orientation[act]

                while loc[2] != act_orientation:
                    loc = step_aware('ACTION:TURNRIGHT', loc, boundaries)
                    entry['dialog'].append(
                        {'id': 'Tourist', 'episode_done': False, 'text': 'ACTION:TURNRIGHT', 'time': t})
                    t += 1

                loc = step_aware('ACTION:FORWARD', loc, boundaries)
                entry['dialog'].append({'id': 'Tourist', 'episode_done': False, 'text': 'ACTION:FORWARD', 'time': t})
                t += 1

                locations.append(loc)

            observations.append(feature_loader.get(config['neighborhood'], loc))

            if len(actions) > T:
                observations = observations[1:]
                actions = actions[1:]
                locations = locations[1:]

        total += 1.
        log.append(entry)

    acc = ((correct / total) * 100)

    return acc, log, numpy.array(num_actions).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--tourist-model', type=str)
    parser.add_argument('--guide-model', type=str)
    parser.add_argument('--communication', type=str, choices=['continuous', 'discrete', 'natural'])
    parser.add_argument('--decoding-strategy', type=str, default='greedy',
                        choices=['beam_search', 'greedy', 'sample'])
    parser.add_argument('--log-name', type=str, default='test')
    parser.add_argument('--T', type=int, default=1)

    args = parser.parse_args()
    print(args)

    # Load data
    train_configs = json.load(open(os.path.join(args.data_dir, 'configurations.train.json')))
    valid_configs = json.load(open(os.path.join(args.data_dir, 'configurations.valid.json')))
    test_configs = json.load(open(os.path.join(args.data_dir, 'configurations.test.json')))

    map = Map(args.data_dir, TalkTheWalkEmergent.neighborhoods)
    feature_loader = GoldstandardFeatures(map)
    dictionary = None

    if args.communication == 'continuous':
        tourist = TouristContinuous.load(args.tourist_model)
        guide = GuideContinuous.load(args.guide_model)
        if args.cuda:
            tourist = tourist.cuda()
            guide = guide.cuda()


        def _predict_location(batch):
            t_out = tourist.forward(batch)
            g_out = guide.forward(t_out, batch)

            return g_out['prob'], None

        T = tourist.T

    elif args.communication == 'discrete':
        tourist = TouristDiscrete.load(args.tourist_model)
        guide = GuideDiscrete.load(args.guide_model)
        if args.cuda:
            tourist = tourist.cuda()
            guide = guide.cuda()
        T = tourist.T

        def _predict_location(batch):
            t_out = tourist(batch)
            if args.cuda:
                t_out['comms'] = [x.cuda() for x in t_out['comms']]
            g_out = guide(t_out['comms'], batch)
            return g_out['prob'], t_out['comms']
    elif args.communication == 'natural':
        tourist = TouristLanguage.load(args.tourist_model)
        guide = GuideLanguage.load(args.guide_model)
        dictionary = Dictionary(os.path.join(args.data_dir, 'dict.txt'), min_freq=0)
        if args.cuda:
            tourist = tourist.cuda()
            guide = guide.cuda()
        T = args.T

        def _predict_location(batch):
            t_out = tourist(batch, train=False, decoding_strategy=args.decoding_strategy)
            batch['utterance'] = t_out['utterance']
            batch['utterance_mask'] = t_out['utterance_mask']
            g_out = guide(batch, add_rl_loss=False)
            return g_out['prob'], batch['utterance']

    collate_fn = get_collate_fn(args.cuda)

    train_acc, train_log, train_num_actions = evaluate(train_configs, _predict_location, collate_fn, map,
                                                       feature_loader, T=T, dict=dictionary,
                                                       communication=args.communication)
    # print(train_acc, train_num_actions)
    # with open('{}.train.json'.format(args.log_name), 'w') as f:
    #     json.dump(train_log, f)
    #
    # valid_acc, valid_log, valid_num_actions = evaluate(valid_configs, _predict_location, collate_fn, map,
    #                                                    feature_loader, T=T, dict=dictionary,
    #                                                    communication=args.communication)
    # print(valid_acc, valid_num_actions)
    # with open('{}.valid.json'.format(args.log_name), 'w') as f:
    #     json.dump(valid_log, f)

    test_acc, test_log, test_num_actions = evaluate(test_configs, _predict_location, collate_fn, map,
                                                    feature_loader, T=T, dict=dictionary,
                                                    communication=args.communication)
    print(test_acc, test_num_actions)
    with open('{}.test.json'.format(args.log_name), 'w') as f:
        json.dump(test_log, f)
