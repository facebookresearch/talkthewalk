import argparse
import os
import json
import random
import torch
import copy
import time
import numpy

from data_loader import create_obs_dict, Map, load_features, FasttextFeatures, GoldstandardFeatures, ResnetFeatures, step_aware
from predict_location_continuous import create_batch, LocationPredictor
from predict_location_discrete import TouristDiscrete, GuideDiscrete


landmarks = {}

def get_landmarks(neighborhood, boundaries):
    if neighborhood not in landmarks:
        landmarks[neighborhood] = json.load(open('./data/{}_map.json'.format(neighborhood)))

    landmark_list = []
    for landmark in landmarks[neighborhood]:
        if boundaries[0] >= landmark['x']*2 <= boundaries[2] and boundaries[1] >= landmark['y'] <= boundaries[3]:
            landmark_list.append(landmark)
    return landmark_list


def evaluate(configs, predict_location_fn, landmark_map, feature_loaders, random_walk=True, cuda=False, T=2):
    correct, total = 0.0, 0.0
    num_actions = []
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

        landmarks, target_index = landmark_map.get_landmarks(config['neighborhood'], boundaries, target_loc)

        flat_target_index = target_index[0]*4 + target_index[1]

        max_eval = 3
        loc = [boundaries[0] + random.randint(0, 3), boundaries[1] + random.randint(0, 3), 0]
        feature_list = {k: [feature_loaders[k].get(config['neighborhood'], loc)] for k in feature_loaders.keys()}
        action_list = []
        locations = [loc]
        predicted = list()

        entry['start_location'] = copy.deepcopy(loc)
        t = time.time()

        for step in range(150):
            if len(action_list) == T:
                features = {k: [v] for k, v in feature_list.items()}
                if len(action_list) == 0:
                    actions = [[0]]
                else:
                    actions = [action_list]

                y_batch = [[locations[0][0]-boundaries[0], locations[0][1] - boundaries[1]]]
                X_batch, action_batch, landmark_batch, y_batch = create_batch(features, actions, [landmarks], y_batch, cuda=cuda)

                prob, t_comms = predict_location_fn(X_batch, action_batch, landmark_batch, y_batch)
                if isinstance(t_comms, torch.FloatTensor):
                    entry['dialog'].append({'id': 'Tourist', 'episode_done': False, 'text': ''.join(['%0.0f' % x for x in t_comms[0].cpu().data.numpy()[0, :]]), 'time': t})
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
                    entry['dialog'].append({'id': 'Guide', 'episode_done': False, 'text': 'EVALUATE_LOCATION', 'time': t})
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
                act = random.randint(0, 3)
                action_list.append(act)

                while loc[2] != act:
                    loc = step_aware(1, loc, boundaries)
                    entry['dialog'].append({'id': 'Tourist', 'episode_done': False, 'text': 'ACTION:TURNRIGHT', 'time': t})
                    t+=1

                loc = step_aware(2, loc, boundaries)
                entry['dialog'].append({'id': 'Tourist', 'episode_done': False, 'text': 'ACTION:FORWARD', 'time': t})
                t+=1

                locations.append(loc)

            for k in feature_loaders.keys():
                feature_list[k].append(feature_loaders[k].get(config['neighborhood'], loc))

            if len(action_list) > T:
                feature_list = {k: feature_list[k][1:] for k in feature_loaders.keys()}
                action_list = action_list[1:]
                locations = locations[1:]

        total += 1.
        log.append(entry)

    acc = ((correct / total) * 100)

    return acc, log, numpy.array(num_actions).mean()


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
    landmark_map = Map(neighborhoods, include_empty_corners=True)

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
        if args.cuda:
            net.cuda()

        def _predict_location(X_batch, action_batch, landmark_batch, y_batch):
            _, _, prob = net.forward(X_batch, action_batch, landmark_batch, y_batch)
            return prob, _
        T = net.T

    else:
        tourist = TouristDiscrete.load(args.tourist_model)
        guide = GuideDiscrete.load(args.guide_model)
        if args.cuda:
            tourist = tourist.cuda()
            guide = guide.cuda()
        T = tourist.T

        def _predict_location(X_batch, action_batch, landmark_batch, y_batch):
            t_comms, t_probs, t_val = tourist(X_batch, action_batch)
            if args.cuda:
                t_comms = [x.cuda() for x in t_comms]
            prob = guide(t_comms, landmark_batch)
            return prob, t_comms

    accs = []
    num_actions = []
    for _ in range(3):
        train_acc, train_log, train_num_actions = evaluate(train_configs, _predict_location, landmark_map, feature_loaders, cuda=args.cuda, T=T)
        accs.append(train_acc)
        num_actions.append(train_num_actions)

    print(numpy.array(accs).mean(), numpy.array(accs).std())
    print(numpy.array(num_actions).mean(), numpy.array(num_actions).std())

    accs = []
    num_actions = []
    for _ in range(3):
        train_acc, train_log, train_num_actions = evaluate(valid_configs, _predict_location, landmark_map,
                                                           feature_loaders, cuda=args.cuda, T=T)
        accs.append(train_acc)
        num_actions.append(train_num_actions)

    print(numpy.array(accs).mean(), numpy.array(accs).std())
    print(numpy.array(num_actions).mean(), numpy.array(num_actions).std())

    accs = []
    num_actions = []
    for _ in range(3):
        train_acc, train_log, train_num_actions = evaluate(test_configs, _predict_location, landmark_map,
                                                           feature_loaders, cuda=args.cuda, T=T)
        accs.append(train_acc)
        num_actions.append(train_num_actions)

    print(numpy.array(accs).mean(), numpy.array(accs).std())
    print(numpy.array(num_actions).mean(), numpy.array(num_actions).std())












