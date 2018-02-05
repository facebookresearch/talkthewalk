import copy
import os
import json
import itertools
from data_loader import load_data_multiple_step, Landmarks, load_features, GoldstandardFeatures

def prediction_upperbound(seq_of_landmarks, landmark_map, neighborhood, min_x, min_y, x, y, actions=None):
    depth = len(seq_of_landmarks)

    if actions is not None:
        assert len(actions) == depth - 1

    paths = list()
    for i in range(4):
        for j in range(4):
            path = dict()
            path['x'] = min_x + i
            path['y'] = min_y + j

            path['seq_of_landmarks'] = [[landmark_map.stoi[t] + 1 for t in landmark_map.landmarks[neighborhood][(path['x'], path['y'])]]]
            paths.append(path)


    for d in range(depth-1):
        new_paths = list()
        for path in paths:
            if actions is None or actions[d] == 0:
                path_N = copy.deepcopy(path)
                path_N['x'] = path['x']
                path_N['y'] = min(path['y'] + 1, min_y + 3)
                path_N['seq_of_landmarks'].append(
                    [landmark_map.stoi[t] + 1 for t in landmark_map.landmarks[neighborhood][(path_N['x'], path_N['y'])]])
                new_paths.append(path_N)

            if actions is None or actions[d] == 1:
                path_S = copy.deepcopy(path)
                path_S['x'] = path['x']
                path_S['y'] = max(path['y'] - 1, min_y)
                path_S['seq_of_landmarks'].append(
                    [landmark_map.stoi[t] + 1 for t in landmark_map.landmarks[neighborhood][(path_S['x'], path_S['y'])]])
                new_paths.append(path_S)

            if actions is None or actions[d] == 2:
                path_E = copy.deepcopy(path)
                path_E['x'] = min(path['x'] + 1, min_x + 3)
                path_E['y'] = path['y']
                path_E['seq_of_landmarks'].append(
                    [landmark_map.stoi[t] + 1 for t in landmark_map.landmarks[neighborhood][(path_E['x'], path_E['y'])]])
                new_paths.append(path_E)

            if actions is None or actions[d] == 3:
                path_W = copy.deepcopy(path)
                path_W['x'] = max(path['x'] - 1, min_x)
                path_W['y'] = path['y']
                path_W['seq_of_landmarks'].append(
                    [landmark_map.stoi[t] + 1 for t in landmark_map.landmarks[neighborhood][(path_W['x'], path_W['y'])]])
                new_paths.append(path_W)
        paths = new_paths

    correct, total = 0.0, 0.0
    for path in paths:
        if all([l1 == l2 for l1, l2 in zip(path['seq_of_landmarks'], seq_of_landmarks)]):
            if path['x'] == x and path['y'] == y:
                correct += 1
            total += 1
    return correct/total

def process(configs, feature_loaders, landmark_map, num_steps):
    correct, cnt = 0, 0

    all_possible_actions = [[]]
    if num_steps > 1:
        action_set = [[0, 1, 2, 3]] * (num_steps - 1)
        all_possible_actions = list(itertools.product(*action_set))

    for config in configs:
        for a in all_possible_actions:
            neighborhood = config['neighborhood']
            x, y = config['target_location'][:2]
            min_x, min_y = config['boundaries'][:2]

            obs = {k: list() for k in feature_loaders.keys()}
            actions = list()
            sampled_x, sampled_y = x, y
            for p in range(num_steps):
                for k, feature_loader in feature_loaders.items():
                    obs[k].append(feature_loader.get(neighborhood, sampled_x, sampled_y))

                if p != num_steps - 1:
                    sampled_act = a[p]
                    actions.append(sampled_act)
                    step = [(0, 1), (0, -1), (1, 0), (-1, 0)][sampled_act]
                    sampled_x = max(min(sampled_x + step[0], min_x + 3), min_x)
                    sampled_y = max(min(sampled_y + step[1], min_y + 3), min_y)

            correct += prediction_upperbound(obs['goldstandard'], landmark_map, neighborhood, min_x, min_y, sampled_x, sampled_y)
            cnt += 1

    return correct/cnt

if __name__ == '__main__':
    data_dir = os.environ.get('TALKTHEWALK_DATADIR', './data')

    train_configs = json.load(open(os.path.join(data_dir, 'configurations.train.json')))
    valid_configs = json.load(open(os.path.join(data_dir, 'configurations.valid.json')))
    test_configs = json.load(open(os.path.join(data_dir, 'configurations.test.json')))

    neighborhoods = ['fidi', 'hellskitchen', 'williamsburg', 'uppereast', 'eastvillage']
    landmark_map = Landmarks(neighborhoods, include_empty_corners=True)

    feature_loaders = dict()
    feature_loaders['goldstandard'] = GoldstandardFeatures(landmark_map)

    for step in range(1, 5):
        train_upp = process(train_configs, feature_loaders, landmark_map, step)
        valid_upp = process(valid_configs, feature_loaders, landmark_map, step)
        test_upp = process(test_configs, feature_loaders, landmark_map, step)

        print("%.2f, %.2f, %.2f" % (train_upp*100, valid_upp*100, test_upp*100))