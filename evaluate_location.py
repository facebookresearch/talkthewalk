import argparse
import os
import json
import random
import torch

from torch.autograd import Variable
from data_loader import create_obs_dict, Landmarks, load_features, FasttextFeatures, GoldstandardFeatures, ResnetFeatures, load_data_multiple_step
from predict_location_emergent import Guide, Tourist
from predict_location_multiple_step import create_batch, LocationPredictor, epoch


def evaluate(configs, tourist, guide, landmark_map, feature_loader):
    correct, total = 0.0, 0.0
    for config in configs:
        target_x = config['target_location'][0]
        target_y = config['target_location'][1]

        landmarks, target_index = landmark_map.get_landmarks(config['neighborhood'], config['boundaries'][0],
                                                              config['boundaries'][1], target_x, target_y)

        max_eval = 3
        for _ in range(50):
            sampled_x, sampled_y = config['boundaries'][0] + random.randint(0, 3), config['boundaries'][
                1] + random.randint(0, 3)

            features = feature_loader.get(config['neighborhood'], sampled_x, sampled_y)
            X, l, mask, y = create_batch([features], [landmarks], [target_index])

            t_comms, t_probs, t_val = tourist(X)
            t_msg = Variable(t_comms.data)
            if args.cuda:
                t_msg = t_msg.cuda()
            g_prob = guide(t_msg, l, mask)

            sampled_index = torch.multinomial(g_prob, 1)
            if sampled_index == target_index:
                if (sampled_x, sampled_y) == (target_x, target_y):
                    correct += 1
                    break
                else:
                    max_eval -= 1
                    if max_eval <= 0:
                        break
        total += 1

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

        train_acc = evaluate(train_configs, tourist, guide, landmark_map, feature_loader)
        valid_acc = evaluate(valid_configs, tourist, guide, landmark_map, feature_loader)
        test_acc = evaluate(test_configs, tourist, guide, landmark_map, feature_loader)

        print(train_acc, valid_acc, test_acc)










