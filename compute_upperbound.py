import copy
import operator
import os
import json
import random
import itertools
from data_loader import Landmarks, GoldstandardFeatures, step_agnostic, step_aware


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

def prediction_upperbound(seq_of_landmarks, goldstandard_features, neighborhood, boundaries, loc, actions=None, num_actions=4, step_fn=step_agnostic):
    depth = len(seq_of_landmarks)

    if actions is not None:
        assert len(actions) == depth - 1

    if num_actions == 4:
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
            for act_index in range(num_actions):
                if actions is None or actions[d] == act_index:
                    path_new = copy.deepcopy(path)
                    path_new['loc'] = step_fn(act_index, path['loc'], boundaries)
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


def process(configs, feature_loaders, num_steps, step_fn, num_actions=4):
    correct, cnt = 0, 0

    all_possible_actions = [[]]
    if num_steps > 1:
        actions = [i for i in range(num_actions)]
        action_set = [actions] * (num_steps - 1)
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

            correct += prediction_upperbound(obs['goldstandard'], feature_loaders['goldstandard'],
                                             neighborhood, boundaries, loc,
                                             step_fn=step_fn, num_actions=num_actions)
            cnt += 1

    return correct/cnt

if __name__ == '__main__':
    data_dir = os.environ.get('TALKTHEWALK_DATADIR', './data')

    train_configs = json.load(open(os.path.join(data_dir, 'configurations.train.json')))
    valid_configs = json.load(open(os.path.join(data_dir, 'configurations.valid.json')))
    test_configs = json.load(open(os.path.join(data_dir, 'configurations.test.json')))

    neighborhoods = ['fidi', 'hellskitchen', 'williamsburg', 'uppereast', 'eastvillage']
    landmark_map = Landmarks(neighborhoods, include_empty_corners=True)

    orientation_aware = False
    if orientation_aware:
        step_fn = step_aware
        num_actions = 3
    else:
        step_fn = step_agnostic
        num_actions = 4

    feature_loaders = dict()
    feature_loaders['goldstandard'] = GoldstandardFeatures(landmark_map, orientation_aware=orientation_aware)

    for step in range(1, 5):
        train_upp = process(train_configs, feature_loaders, step, step_fn, num_actions)
        valid_upp = process(valid_configs, feature_loaders, step, step_fn, num_actions)
        test_upp = process(test_configs, feature_loaders, step, step_fn, num_actions)

        print("%.2f, %.2f, %.2f" % (train_upp*100, valid_upp*100, test_upp*100))