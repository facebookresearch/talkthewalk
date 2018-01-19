import argparse
import json
import random
import torch

from torch.autograd import Variable
from data_loader import create_obs_dict, Landmarks, load_features, TextrecogFeatures, GoldstandardFeatures, create_batch
from predict_location_emergent import Guide, Tourist


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--features', choices=['goldstandard', 'textrecog', 'resnet'], default='textrecog')
    parser.add_argument('--tourist-model', type=str)
    parser.add_argument('--guide-model', type=str)

    args = parser.parse_args()

    # Load data
    neighborhoods = ['fidi', 'hellskitchen', 'williamsburg', 'uppereast', 'eastvillage']
    landmark_map = Landmarks(neighborhoods, include_empty_corners=True)
    feature_loader = None
    if args.features == 'textrecog':
        textfeatures = load_features(neighborhoods)
        obs_i2s, obs_s2i = create_obs_dict(textfeatures, neighborhoods)
        feature_loader = TextrecogFeatures(textfeatures, obs_s2i)
    if args.features == 'goldstandard':
        feature_loader = GoldstandardFeatures(landmark_map)

    assert (feature_loader is not None)

    train_configs = json.load(open('configurations.train.json'))
    valid_configs = json.load(open('configurations.valid.json'))
    test_configs = json.load(open('configurations.test.json'))

    tourist = Tourist.load(args.tourist_model)
    guide = Guide.load(args.guide_model)

    train_acc = evaluate(train_configs, tourist, guide, landmark_map, feature_loader)
    valid_acc = evaluate(valid_configs, tourist, guide, landmark_map, feature_loader)
    test_acc = evaluate(test_configs, tourist, guide, landmark_map, feature_loader)

    print(train_acc, valid_acc, test_acc)










