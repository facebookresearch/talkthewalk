# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import os
import itertools
import json
import numpy

from torch.utils.data.dataset import Dataset
from sklearn.decomposition import PCA

from ttw.dict import Dictionary, LandmarkDictionary, ActionAgnosticDictionary, ActionAwareDictionary, TextrecogDict, \
    START_TOKEN, END_TOKEN
from ttw.env import step_agnostic, step_aware

neighborhoods = ['hellskitchen', 'williamsburg', 'eastvillage', 'fidi', 'uppereast']
boundaries = dict()
boundaries['hellskitchen'] = [3, 3]
boundaries['williamsburg'] = [2, 8]
boundaries['eastvillage'] = [3, 4]
boundaries['fidi'] = [2, 3]
boundaries['uppereast'] = [3, 3]


class TalkTheWalkEmergent(Dataset):
    """Dataset loading for emergent language experiments

    Generates all tourist trajectories of length T"""

    def __init__(self, data_dir, set, goldstandard_features=True, resnet_features=False, fasttext_features=False, T=2):
        self.data_dir = data_dir
        self.map = Map(data_dir, neighborhoods, include_empty_corners=True)
        self.T = T
        self.act_dict = ActionAgnosticDictionary()

        self.configs = json.load(open(os.path.join(data_dir, 'configurations.{}.json'.format(set))))
        self.feature_loaders = dict()
        self.data = {}
        if fasttext_features:
            textfeatures = dict()
            for n in neighborhoods:
                textfeatures[n] = json.load(open(os.path.join(data_dir, n, "text.json")))
            self.feature_loaders['fasttext'] = FasttextFeatures(textfeatures, os.path.join(data_dir, 'wiki.en.bin'))
            self.data['fasttext'] = list()
        if resnet_features:
            self.feature_loaders['resnet'] = ResnetFeatures(os.path.join(data_dir, 'resnetfeat.json'))
            self.data['fasttext'] = list()
        if goldstandard_features:
            self.feature_loaders['goldstandard'] = GoldstandardFeatures(self.map)
            self.data['goldstandard'] = list()
        assert (len(self.feature_loaders) > 0)

        self.data['actions'] = list()
        self.data['landmarks'] = list()
        self.data['target'] = list()

        action_set = [['UP', 'DOWN', 'LEFT', 'RIGHT']] * self.T
        all_possible_actions = list(itertools.product(*action_set))

        for config in self.configs:
            for a in all_possible_actions:
                neighborhood = config['neighborhood']
                target_loc = config['target_location']
                boundaries = config['boundaries']

                obs = {k: list() for k in self.feature_loaders.keys()}
                actions = list()
                loc = copy.deepcopy(config['target_location'])
                for p in range(self.T + 1):
                    for k, feature_loader in self.feature_loaders.items():
                        obs[k].append(feature_loader.get(neighborhood, loc))

                    if p != self.T:
                        sampled_act = self.act_dict.encode(a[p])
                        actions.append(sampled_act)
                        loc = step_agnostic(a[p], loc, boundaries)

                if self.T == 0:
                    actions.append(0)

                for k in self.feature_loaders.keys():
                    self.data[k].append(obs[k])

                self.data['actions'].append(actions)
                landmarks, label_index = self.map.get_landmarks(neighborhood, boundaries, target_loc)
                self.data['landmarks'].append(landmarks)
                self.data['target'].append(label_index)

    def __getitem__(self, index):
        return {key: self.data[key][index] for key in self.data.keys()}

    def __len__(self):
        return len(self.data['actions'])


class TalkTheWalkLanguage(Dataset):
    """Dataset loading for natural language experiments.

    Only contains trajectories taken by human annotators
    """

    def __init__(self, data_dir, set, last_turns=1, min_freq=3, min_sent_len=2, orientation_aware=False,
                 include_guide_utterances=True):
        self.dialogues = json.load(open(os.path.join(data_dir, 'talkthewalk.{}.json'.format(set))))
        self.dict = Dictionary(file=os.path.join(data_dir, 'dict.txt'), min_freq=min_freq)
        self.map = Map(data_dir, neighborhoods, include_empty_corners=True)
        self.act_dict = ActionAgnosticDictionary()
        self.act_aware_dict = ActionAwareDictionary()

        self.feature_loader = GoldstandardFeatures(self.map)

        self.data = dict()
        self.data['actions'] = list()
        self.data['goldstandard'] = list()
        self.data['landmarks'] = list()
        self.data['target'] = list()
        self.data['utterance'] = list()

        for config in self.dialogues:
            loc = config['start_location']
            neighborhood = config['neighborhood']
            boundaries = config['boundaries']
            act_memory = list()
            obs_memory = [self.feature_loader.get(neighborhood, loc)]

            dialogue_context = list()
            for msg in config['dialog']:
                if msg['id'] == 'Tourist':
                    act = msg['text']
                    act_id = self.act_aware_dict.encode(act)
                    if act_id >= 0:
                        new_loc = step_aware(act, loc, boundaries)
                        old_loc = loc
                        loc = new_loc

                        if orientation_aware:
                            act_memory.append(act_id)
                            obs_memory.append(self.feature_loader.get(neighborhood, new_loc))
                        else:
                            if act == 'ACTION:FORWARD':  # went forward
                                act_dir = self.act_dict.encode_from_location(old_loc, new_loc)
                                act_memory.append(act_dir)
                                obs_memory.append(self.feature_loader.get(neighborhood, loc))
                    elif len(msg['text'].split(' ')) > min_sent_len:
                        dialogue_context.append(self.dict.encode(msg['text']))
                        utt = self.dict.encode(START_TOKEN) + [y for x in dialogue_context[-last_turns:] for y in x] \
                              + self.dict.encode(END_TOKEN)
                        self.data['utterance'].append(utt)

                        landmarks, tgt = self.map.get_landmarks(config['neighborhood'], boundaries, loc)
                        self.data['landmarks'].append(landmarks)
                        self.data['target'].append(tgt)

                        self.data['actions'].append(act_memory)
                        self.data['goldstandard'].append(obs_memory)

                        act_memory = list()
                        obs_memory = [self.feature_loader.get(neighborhood, loc)]
                elif include_guide_utterances:
                    dialogue_context.append(self.dict.encode(msg['text']))

    def __getitem__(self, index):
        return {key: self.data[key][index] for key in self.data.keys()}

    def __len__(self):
        return len(self.data['target'])


class TalkTheWalkLandmarks(Dataset):
    """Creates dataset for landmark classification"""

    def __init__(self, data_dir, resnet_features, fasttext_features, textrecog_features, n_components=100, pca=False):
        self.feature_loaders = dict()
        self.num_tokens = None
        self.map = Map(data_dir, neighborhoods)
        if fasttext_features or textrecog_features:
            self.textfeatures = dict()
            for n in neighborhoods:
                self.textfeatures[n] = json.load(open(os.path.join(data_dir, n, "text.json")))
            self.textrecog_dict = TextrecogDict(self.textfeatures)

        if fasttext_features:
            self.feature_loaders['fasttext'] = FasttextFeatures(self.textfeatures,
                                                                os.path.join(data_dir, 'wiki.en.bin'),
                                                                pca=pca,
                                                                n_components=n_components)
        if resnet_features:
            self.feature_loaders['resnet'] = ResnetFeatures(os.path.join(data_dir, 'resnetfeat.json'),
                                                            pca=pca,
                                                            n_components=n_components)
        if textrecog_features:
            self.feature_loaders['textrecog'] = TextrecogFeatures(self.textfeatures, self.textrecog_dict)
            self.num_tokens = len(self.textrecog_dict)

        assert (len(self.feature_loaders) > 0)

        self.data = {'target': list()}
        for k in self.feature_loaders.keys():
            self.data[k] = list()

        for n in neighborhoods:
            for x, tmp in enumerate(self.map.coord_to_landmarks[n]):
                for y, ls in enumerate(tmp):
                    target = [0] * 10
                    for k in self.feature_loaders.keys():
                        self.data[k].append(self.feature_loaders[k].get(n, [x, y, 0]))
                    if len(ls) > 0:
                        for l in ls:
                            target[l - 1] = 1
                    else:
                        target[self.map.landmark_dict.encode('Empty') - 1] = 1
                    self.data['target'].append(target)

    def __len__(self):
        return len(self.data['target'])

    def __getitem__(self, index):
        return {k: self.data[k][index] for k in self.data.keys()}


class DatasetHolder(Dataset):
    """Constructs PyTorch compatible dataset from dictionary"""
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return {k: self.data[k][index] for k in self.data.keys()}

    def __len__(self):
        return len(self.data['target'])


class Map(object):
    """Map with landmarks"""

    def __init__(self, data_dir, neighborhoods, include_empty_corners=True):
        super(Map, self).__init__()
        self.coord_to_landmarks = dict()
        self.include_empty_corners = include_empty_corners
        self.landmark_dict = LandmarkDictionary()
        self.data_dir = data_dir
        self.landmarks = dict()

        for neighborhood in neighborhoods:
            self.coord_to_landmarks[neighborhood] = [[[] for _ in range(boundaries[neighborhood][1] * 2 + 4)]
                                                     for _ in range(boundaries[neighborhood][0] * 2 + 4)]
            self.landmarks[neighborhood] = json.load(open(os.path.join(data_dir, neighborhood, "map.json")))
            for landmark in self.landmarks[neighborhood]:
                coord = self.transform_map_coordinates(landmark)
                landmark_idx = self.landmark_dict.encode(landmark['type'])
                self.coord_to_landmarks[neighborhood][coord[0]][coord[1]].append(landmark_idx)

    def transform_map_coordinates(self, landmark):
        x_offset = {"NW": 0, "SW": 0, "NE": 1, "SE": 1}
        y_offset = {"NW": 1, "SW": 0, "NE": 1, "SE": 0}

        coord = (landmark['x'] * 2 + x_offset[landmark['orientation']],
                 landmark['y'] * 2 + y_offset[landmark['orientation']])
        return coord

    def get(self, neighborhood, x, y):
        landmarks = self.coord_to_landmarks[neighborhood][x][y]
        if self.include_empty_corners and len(landmarks) == 0:
            return [self.landmark_dict.encode('Empty')]
        return landmarks

    def get_landmarks(self, neighborhood, boundaries, target_loc):
        landmarks = [[[] for _ in range(4)] for _ in range(4)]
        label_index = (target_loc[0] - boundaries[0], target_loc[1] - boundaries[1])
        for x in range(4):
            for y in range(4):
                landmarks[x][y] = self.get(neighborhood, boundaries[0] + x, boundaries[1] + y)

        assert 0 <= label_index[0] < 4
        assert 0 <= label_index[1] < 4

        return landmarks, label_index

    def get_unprocessed_landmarks(self, neighborhood, boundaries):
        landmark_list = []
        for landmark in self.landmarks[neighborhood]:
            coord = self.transform_map_coordinates(landmark)
            if boundaries[0] <= coord[0] <= boundaries[2] and boundaries[1] <= coord[1] <= boundaries[3]:
                landmark_list.append(landmark)
        return landmark_list


class GoldstandardFeatures:
    def __init__(self, map, orientation_aware=False):
        self.map = map
        self.allowed_orientations = {'NW': [3, 0], 'SW': [2, 3], 'NE': [0, 1], 'SE': [1, 2]}
        self.mod2orientation = {(0, 0): 'SW', (1, 0): 'SE', (0, 1): 'NW', (1, 1): 'NE'}
        self.orientation_aware = orientation_aware

    def get(self, neighborhood, loc):
        if self.orientation_aware:
            mod = (loc[0] % 2, loc[1] % 2)
            orientation = self.mod2orientation[mod]
            if loc[2] in self.allowed_orientations[orientation]:
                return self.map.get(neighborhood, loc[0], loc[1])
            else:
                return [self.map.encode('Empty')]
        else:
            return self.map.get(neighborhood, loc[0], loc[1])


class TextrecogFeatures:
    def __init__(self, textfeatures, textrecog_dict):
        self.textfeatures = textfeatures
        self.textrecog_dict = textrecog_dict

    def get(self, neighborhood, loc):
        obs = list()
        for key in get_orientation_keys(loc[0], loc[1]):
            obs.extend([self.textrecog_dict.encode(o['lex_recog']) for o in self.textfeatures[neighborhood][key]])
        return obs


class FasttextFeatures:
    def __init__(self, textfeatures, fasttext_file, pca=False, n_components=100):
        import fastText
        self.textfeatures = textfeatures
        self.f = fastText.load_model(fasttext_file)
        self.pca = pca

        self.i2k = list()
        self.k2i = dict()
        self.X = list()
        if pca:
            for n in self.textfeatures.keys():
                for k in self.textfeatures[n].keys():
                    for o in self.textfeatures[n][k]:
                        if o['lex_recog'] not in self.k2i:
                            self.X.append(self.f.get_word_vector(o['lex_recog']))
                            self.i2k.append(o['lex_recog'])
                            self.k2i[o['lex_recog']] = len(self.i2k)
            self.X = numpy.array(self.X)
            self.X = PCA(n_components=n_components).fit_transform(self.X)

    def get(self, neighborhood, loc):
        obs = list()
        for key in get_orientation_keys(loc[0], loc[1]):
            if self.pca:
                obs.extend([self.X[self.k2i[o['lex_recog']]].tolist() for o in self.textfeatures[neighborhood][key]])
            else:
                obs.extend(
                    [self.f.get_word_vector(o['lex_recog']).tolist() for o in self.textfeatures[neighborhood][key]])

        return obs


class ResnetFeatures:
    def __init__(self, file, pca=False, n_components=512):
        self.resnetfeatures = json.load(open(file))
        self.pca = pca
        self.i2k = list()
        self.k2i = dict()
        self.X = list()
        if pca:
            for n in self.resnetfeatures.keys():
                for k in self.resnetfeatures[n].keys():
                    self.X.append(self.resnetfeatures[n][k])
                    complete_key = "{}_{}".format(n, k)
                    self.i2k.append(complete_key)
                    self.k2i[complete_key] = len(self.i2k)
            self.X = numpy.array(self.X)
            self.X = PCA(n_components=n_components).fit_transform(self.X)

    def get(self, neighborhood, loc):
        obs = list()
        for key in get_orientation_keys(loc[0], loc[1]):
            if self.pca:
                obs.append(self.X[self.k2i["{}_{}".format(neighborhood, key)]])
            else:
                obs.append(self.resnetfeatures[neighborhood][key])
        return obs


def get_orientation_keys(x, y, cross_the_street=False):
    """Get orientations at location (x,y) st you're facing landmarks.
    """
    if x % 2 == 0 and y % 2 == 0:
        k = ["{}_{}_{}".format(x, y, 'W1'),
             "{}_{}_{}".format(x, y, 'W2'),
             "{}_{}_{}".format(x, y, 'S1'),
             "{}_{}_{}".format(x, y, 'S2')]
        if cross_the_street:
            k += ["{}_{}_{}".format(x + 1, y, 'W1'),
                  "{}_{}_{}".format(x + 1, y, 'W2'),
                  "{}_{}_{}".format(x, y + 1, 'S1'),
                  "{}_{}_{}".format(x, y + 1, 'S2')]
        return k
    if x % 2 == 1 and y % 2 == 0:
        k = ["{}_{}_{}".format(x, y, 'E1'),
             "{}_{}_{}".format(x, y, 'E2'),
             "{}_{}_{}".format(x, y, 'S1'),
             "{}_{}_{}".format(x, y, 'S2')]
        if cross_the_street:
            k += ["{}_{}_{}".format(x - 1, y, 'E1'),
                  "{}_{}_{}".format(x - 1, y, 'E2'),
                  "{}_{}_{}".format(x, y + 1, 'S1'),
                  "{}_{}_{}".format(x, y + 1, 'S2')]
        return k
    if x % 2 == 0 and y % 2 == 1:
        k = ["{}_{}_{}".format(x, y, 'W1'),
             "{}_{}_{}".format(x, y, 'W2'),
             "{}_{}_{}".format(x, y, 'N1'),
             "{}_{}_{}".format(x, y, 'N2')]
        if cross_the_street:
            k += ["{}_{}_{}".format(x + 1, y, 'W1'),
                  "{}_{}_{}".format(x + 1, y, 'W2'),
                  "{}_{}_{}".format(x, y - 1, 'N1'),
                  "{}_{}_{}".format(x, y - 1, 'N2')]
        return k
    if x % 2 == 1 and y % 2 == 1:
        k = ["{}_{}_{}".format(x, y, 'E1'),
             "{}_{}_{}".format(x, y, 'E2'),
             "{}_{}_{}".format(x, y, 'N1'),
             "{}_{}_{}".format(x, y, 'N2')]
        if cross_the_street:
            k += ["{}_{}_{}".format(x - 1, y, 'E1'),
                  "{}_{}_{}".format(x - 1, y, 'E2'),
                  "{}_{}_{}".format(x, y - 1, 'N1'),
                  "{}_{}_{}".format(x, y - 1, 'N2')]
        return k
