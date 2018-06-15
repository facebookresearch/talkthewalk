# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""This file contains code for generating the train, valid, and test split for TTW."""
import json
import argparse

from ttw.data_loader import neighborhoods, boundaries

key2set = dict()

def get_configurations(neighborhoods):
    train_configurations = list()
    valid_configurations = list()
    test_configurations = list()

    for neighborhood in neighborhoods:
        cnt = 0
        for minimum_x in range(boundaries[neighborhood][0]+1):
            for minimum_y in range(boundaries[neighborhood][1]+1):
                min_x = minimum_x*2
                min_y = minimum_y*2

                boundary_config = list()
                for i in range(4):
                    for j in range(4):
                        x = min_x + i
                        y = min_y + j

                        config = {'neighborhood': neighborhood,
                                  'target_location': [x, y, 0],
                                  'boundaries': [min_x, min_y, min_x + 3, min_y + 3]}

                        boundary_config.append(config)

                        if not (minimum_y >= boundaries[neighborhood][1]-1 and minimum_x < 3):
                            key2set[(neighborhood, minimum_x, minimum_y)] = 'train'
                            train_configurations.append(config)
                            cnt += 1
                        elif minimum_y == boundaries[neighborhood][1] and minimum_x < 2:
                            key2set[(neighborhood, minimum_x, minimum_y)] = 'test'
                            test_configurations.append(config)
                            cnt += 1
                        else:
                            key2set[(neighborhood, minimum_x, minimum_y)] = 'valid'
                            valid_configurations.append(config)
                            cnt += 1
    return train_configurations, valid_configurations, test_configurations

train_configurations, valid_configurations, test_configurations = get_configurations(neighborhoods)

print(len(train_configurations), len(valid_configurations), len(test_configurations))

with open('configurations.train.json', 'w') as f:
    json.dump(train_configurations, f)

with open('configurations.valid.json', 'w') as f:
    json.dump(valid_configurations, f)

with open('configurations.test.json', 'w') as f:
    json.dump(test_configurations, f)


train_dialogues = list()
valid_dialogues = list()
test_dialogues = list()

data = json.load(open('./data/talkthewalk.train.json'))
data.extend(json.load(open('./data/talkthewalk.valid.json')))
data.extend(json.load(open('./data/talkthewalk.test.json')))

not_included = dict()

for dialogue in data:
    key = (dialogue['neighborhood'], dialogue['boundaries'][0]//2, dialogue['boundaries'][1]//2)

    if key2set.get(key) == 'train':
        train_dialogues.append(dialogue)
    elif key2set.get(key) == 'valid':
        valid_dialogues.append(dialogue)
    elif key2set.get(key) == 'test':
        test_dialogues.append(dialogue)
    else:
        if key not in not_included:
            not_included[key] = True

with open('talkthewalk.train.json', 'w') as f:
    json.dump(train_dialogues, f)

with open('talkthewalk.valid.json', 'w') as f:
    json.dump(valid_dialogues, f)

with open('talkthewalk.test.json', 'w') as f:
    json.dump(test_dialogues, f)
