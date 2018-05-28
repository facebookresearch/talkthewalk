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


def evaluate(configs, predict_location_fn, collate_fn, map, feature_loader, random_walk=True, cuda=False, T=2):
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
                if isinstance(t_comms, torch.FloatTensor):
                    entry['dialog'].append({'id': 'Tourist', 'episode_done': False,
                                            'text': ''.join(['%0.0f' % x for x in t_comms[0].cpu().data.numpy()[0, :]]),
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

    args = parser.parse_args()
    print(args)

    # Load data
    train_configs = json.load(open(os.path.join(args.data_dir, 'configurations.train.json')))
    valid_configs = json.load(open(os.path.join(args.data_dir, 'configurations.valid.json')))
    test_configs = json.load(open(os.path.join(args.data_dir, 'configurations.test.json')))

    map = Map(args.data_dir, TalkTheWalkEmergent.neighborhoods)
    feature_loader = GoldstandardFeatures(map)

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

    collate_fn = get_collate_fn(args.cuda)
    accs = []
    num_actions = []
    for _ in range(1):
        train_acc, train_log, train_num_actions = evaluate(train_configs, _predict_location, collate_fn, map,
                                                           feature_loader, cuda=args.cuda, T=T)
        accs.append(train_acc)
        num_actions.append(train_num_actions)

    print(numpy.array(accs).mean(), numpy.array(accs).std())
    print(numpy.array(num_actions).mean(), numpy.array(num_actions).std())

    accs = []
    num_actions = []
    for _ in range(1):
        train_acc, train_log, train_num_actions = evaluate(valid_configs, _predict_location, collate_fn, map,
                                                           feature_loader, cuda=args.cuda, T=T)
        accs.append(train_acc)
        num_actions.append(train_num_actions)

    print(numpy.array(accs).mean(), numpy.array(accs).std())
    print(numpy.array(num_actions).mean(), numpy.array(num_actions).std())

    accs = []
    num_actions = []
    for _ in range(1):
        train_acc, train_log, train_num_actions = evaluate(test_configs, _predict_location, collate_fn, map,
                                                           feature_loader, cuda=args.cuda, T=T)
        accs.append(train_acc)
        num_actions.append(train_num_actions)

    print(numpy.array(accs).mean(), numpy.array(accs).std())
    print(numpy.array(num_actions).mean(), numpy.array(num_actions).std())
