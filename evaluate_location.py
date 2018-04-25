import argparse
import os
import json
import random
import torch
import copy
import time

from torch.autograd import Variable
from data_loader import create_obs_dict, Landmarks, load_features, FasttextFeatures, GoldstandardFeatures, ResnetFeatures, load_data_multiple_step, step_agnostic, step_aware
from predict_location_continuous import create_batch, LocationPredictor
from predict_location_discrete import Tourist, Guide


def evaluate(configs, tourist, guide, landmark_map, feature_loaders, random_walk=True):
    tourist.cuda()
    guide.cuda()
    max_steps = tourist.max_steps
    print(max_steps)
    correct, total = 0.0, 0.0

    for config in configs:
        neighborhood = config['neighborhood']
        boundaries = config['boundaries']
        target_loc = config['target_location']

        landmarks, target_index = landmark_map.get_landmarks_2d(config['neighborhood'], boundaries, target_loc)

        flat_target_index = target_index[0]*4 + target_index[1]

        max_eval = 3
        loc = [boundaries[0] + random.randint(0, 3), boundaries[1] + random.randint(0, 3), 0]
        feature_list = {k: [feature_loaders[k].get(config['neighborhood'], loc)] for k in feature_loaders.keys()}
        action_list = []
        locations = [loc]
        predicted = list()


        for _ in range(150):
            if len(action_list) == max_steps - 1:
                features = {k: [v] for k, v in feature_list.items()}
                if len(action_list) == 0:
                    actions = [[0]]
                else:
                    actions = [action_list]

                y_batch = [[locations[0][0]-boundaries[0], locations[0][1] - boundaries[1]]]
                X_batch, action_batch, landmark_batch, y_batch = create_batch(features, actions, [landmarks], y_batch, cuda=True)

                t_comms, t_probs, t_val = tourist(X_batch, action_batch)
                prob = guide(t_comms, landmark_batch)


                sampled_index = torch.multinomial(prob, 1)
                if sampled_index == flat_target_index:
                    predicted.append(locations[0])
                    if locations[0][0] == target_loc[0] and locations[0][1] == target_loc[1]:
                        correct += 1
                        break
                    else:
                        max_eval -= 1
                        if max_eval <= 0:
                            break

            if random_walk:
                act = random.randint(0, 3)
                action_list.append(act)
                loc = step_agnostic(act, loc, boundaries)
                locations.append(loc)
            else:
                sampled_x, sampled_y = config['boundaries'][0] + random.randint(0, 3), config['boundaries'][
                    1] + random.randint(0, 3)

            for k in feature_loaders.keys():
                feature_list[k].append(feature_loaders[k].get(config['neighborhood'], loc))

            if len(action_list) > max_steps - 1:
                feature_list = {k: feature_list[k][1:] for k in feature_loaders.keys()}
                action_list = action_list[1:]
                locations = locations[1:]

        total += 1.

    return ((correct / total) * 100)

landmarks = {}

def get_landmarks(neighborhood, boundaries):
    if neighborhood not in landmarks:
        landmarks[neighborhood] = json.load(open('./data/{}_map.json'.format(neighborhood)))

    landmark_list = []
    for landmark in landmarks[neighborhood]:
        if boundaries[0] >= landmark['x']*2 <= boundaries[2] and boundaries[1] >= landmark['y'] <= boundaries[3]:
            landmark_list.append(landmark)
    return landmark_list


def evaluate_and_log(configs, tourist, guide, landmark_map, feature_loaders, random_walk=True, log_file='log.json'):
    tourist.cuda()
    guide.cuda()
    print(tourist.emb_sz)
    print(tourist.vocab_sz)
    print(guide.embed_sz)

    max_steps = tourist.max_steps
    correct, total = 0.0, 0.0
    log = []

    for config in configs:
        neighborhood = config['neighborhood']
        boundaries = config['boundaries']
        target_loc = config['target_location']
        entry = {'neighborhood': neighborhood,
                 'boundaries': boundaries,
                 'target_location': target_loc,
                 'landmarks': get_landmarks(neighborhood, boundaries),
                 'dialog': []}

        landmarks, target_index = landmark_map.get_landmarks_2d(config['neighborhood'], boundaries, target_loc)

        flat_target_index = target_index[0]*4 + target_index[1]

        max_eval = 3
        loc = [boundaries[0] + random.randint(0, 3), boundaries[1] + random.randint(0, 3), 0]
        feature_list = {k: [feature_loaders[k].get(config['neighborhood'], loc)] for k in feature_loaders.keys()}
        action_list = []
        locations = [loc]
        predicted = list()

        entry['start_location'] = copy.deepcopy(loc)
        t = time.time()

        for _ in range(150):
            if len(action_list) == max_steps - 1:
                features = {k: [v] for k, v in feature_list.items()}
                if len(action_list) == 0:
                    actions = [[0]]
                else:
                    actions = [action_list]

                y_batch = [[locations[0][0]-boundaries[0], locations[0][1] - boundaries[1]]]
                X_batch, action_batch, landmark_batch, y_batch = create_batch(features, actions, [landmarks], y_batch, cuda=True)

                t_comms, t_probs, t_val = tourist(X_batch, action_batch)
                prob = guide(t_comms, landmark_batch)
                entry['dialog'].append({'id': 'Tourist', 'episode_done': False, 'text': ''.join(['%0.0f' % x for x in t_comms[0].cpu().data.numpy()[0, :]]), 'time': t})
                t += 1

                prob_array = [[0 for _ in range(4)] for _ in range(4)]
                prob_data = prob.squeeze().cpu().data.numpy()

                for i in range(prob_data.shape[0]):
                    prob_array[i // 4][i % 4] = float(prob_data[i])

                entry['dialog'].append({'id': 'Guide', 'episode_done': False, 'text': prob_array})
                t += 1

                sampled_index = torch.multinomial(prob, 1)
                if sampled_index == flat_target_index:
                    entry['dialog'].append({'id': 'Guide', 'episode_done': False, 'text': 'EVALUATE_LOCATION', 'time': t})
                    t += 1
                    predicted.append(locations[0])
                    if locations[0][0] == target_loc[0] and locations[0][1] == target_loc[1]:
                        correct += 1
                        break
                    else:
                        max_eval -= 1
                        if max_eval <= 0:
                            break

            if random_walk:
                act = random.randint(0, 3)
                action_list.append(act)

                while loc[2] != act:
                    loc = step_aware(1, loc, boundaries)
                    entry['dialog'].append({'id': 'Tourist', 'episode_done': False, 'text': 'ACTION:TURNRIGHT', 'time': t})
                    t+=1

                loc = step_aware(2, loc, boundaries)
                entry['dialog'].append({'id': 'Tourist', 'episode_done': False, 'text': 'ACTION:FORWARD', 'time': t})
                t+=1

                # loc = step_agnostic(act, loc, boundaries)
                locations.append(loc)
            else:
                sampled_x, sampled_y = config['boundaries'][0] + random.randint(0, 3), config['boundaries'][
                    1] + random.randint(0, 3)

            for k in feature_loaders.keys():
                feature_list[k].append(feature_loaders[k].get(config['neighborhood'], loc))

            if len(action_list) > max_steps - 1:
                feature_list = {k: feature_list[k][1:] for k in feature_loaders.keys()}
                action_list = action_list[1:]
                locations = locations[1:]

        total += 1.
        log.append(entry)


    with open(log_file, 'w') as f:
        json.dump(log, f)

    return ((correct / total) * 100)


def evaluate_location_predictor(configs, net, landmark_map, feature_loaders, random_walk=True):
    net.cuda()
    max_steps = net.max_steps
    correct, total = 0.0, 0.0
    accuracy = 0.0

    for config in configs:
        neighborhood = config['neighborhood']
        min_x = config['boundaries'][0]
        min_y = config['boundaries'][1]
        target_x = config['target_location'][0]
        target_y = config['target_location'][1]

        landmarks, target_index = landmark_map.get_landmarks_2d(config['neighborhood'], min_x,
                                                                min_y, target_x, target_y)

        flat_target_index = target_index[0]*4 + target_index[1]

        max_eval = 3
        sampled_x, sampled_y = min_x + random.randint(0, 3), min_y + random.randint(0, 3)
        steps = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        feature_list = {k: [feature_loaders[k].get(config['neighborhood'], sampled_x, sampled_y)] for k in feature_loaders.keys()}
        action_list = []
        locations = [(sampled_x, sampled_y)]
        predicted = list()

        # obs = {k: list() for k in feature_loaders.keys()}
        # actions = list()
        # sampled_x, sampled_y = target_x, target_y
        # for p in range(max_steps):
        #     for k, feature_loader in feature_loaders.items():
        #         obs[k].append(feature_loader.get(neighborhood, sampled_x, sampled_y))
        #
        #     if p != max_steps - 1:
        #         sampled_act = random.randint(0, 3)
        #         actions.append(sampled_act)
        #         step = [(0, 1), (0, -1), (1, 0), (-1, 0)][sampled_act]
        #         sampled_x = max(min(sampled_x + step[0], min_x + 3), min_x)
        #         sampled_y = max(min(sampled_y + step[1], min_y + 3), min_y)
        #
        # X = {k: [v] for k, v in obs.items()}
        # X_batch, action_batch, landmark_batch, y_batch = create_batch(X, [actions], [landmarks], [target_index],
        #                                                               cuda=True)
        # _, acc, prob = net.forward(X_batch, action_batch, landmark_batch, y_batch)
        # accuracy += acc


        for _ in range(150):
            if len(action_list) == max_steps - 1:
                features = {k: [v] for k, v in feature_list.items()}
                if len(action_list) == 0:
                    actions = [[0]]
                else:
                    actions = [action_list]

                y_batch = [[locations[0][0]-min_x, locations[0][1] - min_y]]
                X_batch, action_batch, landmark_batch, y_batch = create_batch(features, actions, [landmarks], y_batch, cuda=True)

                _, acc, prob = net.forward(X_batch, action_batch, landmark_batch, y_batch)
                accuracy += acc

                sampled_index = torch.multinomial(prob, 1)
                if sampled_index == flat_target_index:
                    predicted.append(locations[0])
                    if locations[0] == (target_x, target_y):
                        correct += 1
                        break
                    else:
                        max_eval -= 1
                        if max_eval <= 0:
                            break

            if random_walk:
                act = random.randint(0, 3)
                action_list.append(act)
                sampled_x = min(max(sampled_x + steps[act][0], config['boundaries'][0]), config['boundaries'][0] + 3)
                sampled_y = min(max(sampled_y + steps[act][1], config['boundaries'][1]), config['boundaries'][1] + 3)
                locations.append((sampled_x, sampled_y))
            else:
                sampled_x, sampled_y = config['boundaries'][0] + random.randint(0, 3), config['boundaries'][
                    1] + random.randint(0, 3)

            for k in feature_loaders.keys():
                feature_list[k].append(feature_loaders[k].get(config['neighborhood'], sampled_x, sampled_y))

            if len(action_list) > max_steps - 1:
                feature_list = {k: feature_list[k][1:] for k in feature_loaders.keys()}
                action_list = action_list[1:]
                locations = locations[1:]

        total += 1.

    return ((correct / total) * 100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--resnet-features', action='store_true')
    parser.add_argument('--fasttext-features', action='store_true')
    parser.add_argument('--goldstandard-features', action='store_true')
    parser.add_argument('--tourist-model', type=str)
    parser.add_argument('--guide-model', type=str)
    parser.add_argument('--predloc-model', type=str)

    args = parser.parse_args()
    print(args)
    # Load data
    neighborhoods = ['fidi', 'hellskitchen', 'williamsburg', 'uppereast', 'eastvillage']
    landmark_map = Landmarks(neighborhoods, include_empty_corners=True)
    print(landmark_map.itos)

    data_dir = './data'
    feature_loaders = dict()
    if args.fasttext_features:
        textfeatures = load_features(neighborhoods)
        obs_i2s, obs_s2i = create_obs_dict(textfeatures, neighborhoods)
        feature_loaders['fasttext'] = FasttextFeatures(textfeatures, '/private/home/harm/data/wiki.en.bin')
    if args.resnet_features:
        feature_loaders['resnet'] = ResnetFeatures(os.path.join(data_dir, 'resnetfeat.json'))
    if args.goldstandard_features:
        feature_loaders['goldstandard'] = GoldstandardFeatures(landmark_map)
    assert (len(feature_loaders) > 0)


    train_configs = json.load(open(os.path.join(data_dir, 'configurations.train.json')))
    valid_configs = json.load(open(os.path.join(data_dir, 'configurations.valid.json')))
    test_configs = json.load(open(os.path.join(data_dir, 'configurations.test.json')))

    if args.predloc_model is not None:
        net = LocationPredictor.load(args.predloc_model)

        train_acc = evaluate_location_predictor(train_configs, net, landmark_map, feature_loaders)
        valid_acc = evaluate_location_predictor(valid_configs, net, landmark_map, feature_loaders)
        test_acc = evaluate_location_predictor(test_configs, net, landmark_map, feature_loaders)
        print(train_acc, valid_acc, test_acc)
    else:
        tourist = Tourist.load(args.tourist_model)
        guide = Guide.load(args.guide_model)

        # train_acc = evaluate_and_log(train_configs, tourist, guide, landmark_map, feature_loaders)
        # valid_acc = evaluate(valid_configs, tourist, guide, landmark_map, feature_loaders)
        # print(valid_acc)
        test_acc = evaluate_and_log(test_configs, tourist, guide, landmark_map, feature_loaders)
        print(test_acc)
        # print(train_acc, valid_acc, test_acc)










