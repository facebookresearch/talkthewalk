import copy
import os
import itertools
import json
import numpy
import torch
import torch.nn.functional as F

from torch.autograd import Variable
from sklearn.decomposition import PCA

from ttw.dict import Dictionary, START_TOKEN, END_TOKEN


class TalkTheWalkEmergent(object):
    """Dataset loading for emergent language experiments"""

    neighborhoods = ['hellskitchen', 'williamsburg', 'eastvillage', 'fidi', 'uppereast']

    def __init__(self, data_dir, set, goldstandard_features=True, resnet_features=False, fasttext_features=False, T=2):
        self.data_dir = data_dir
        self.map = Map(data_dir, TalkTheWalkEmergent.neighborhoods, include_empty_corners=True)
        self.T = T
        self.act_dict = ActionAgnosticDictionary()

        self.configs = json.load(open(os.path.join(data_dir, 'configurations.{}.json'.format(set))))
        self.feature_loaders = dict()
        self.data = {}
        if fasttext_features:
            textfeatures = load_features(data_dir, TalkTheWalkEmergent.neighborhoods)
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


class TalkTheWalkLanguage(object):
    """Dataset loading for natural language experiments"""

    neighborhoods = ['hellskitchen', 'williamsburg', 'eastvillage', 'fidi', 'uppereast']

    def __init__(self, data_dir, set, last_turns=1, min_freq=3, min_sent_len=2, orientation_aware=False, include_guide_utterances=True):
        self.dialogues = json.load(open(os.path.join(data_dir, 'talkthewalk.{}.json'.format(set))))
        self.dict = Dictionary(file=os.path.join(data_dir, 'dict.txt'), min_freq=min_freq)
        self.map = Map(data_dir, TalkTheWalkLanguage.neighborhoods, include_empty_corners=True)
        self.act_dict = ActionAgnosticDictionary()
        self.act_aware_dict = ActionAwareDictionary()

        self.feature_loader = GoldstandardFeatures(self.map)

        self.data = dict()
        self.data['actions'] = list()
        self.data['observations'] = list()
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
                        new_loc = step_aware(act_id, loc, boundaries)
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
                        utt = self.dict.encode(START_TOKEN) + [y for x in dialogue_context[-last_turns:] for y in x]\
                              + self.dict.encode(END_TOKEN)
                        self.data['utterance'].append(utt)

                        landmarks, tgt = self.map.get_landmarks(config['neighborhood'], boundaries, loc)
                        self.data['landmarks'].append(landmarks)
                        self.data['target'].append(tgt)

                        self.data['actions'].append(act_memory)
                        self.data['observations'].append(obs_memory)

                        act_memory = list()
                        obs_memory = [self.feature_loader.get(neighborhood, loc)]
                elif include_guide_utterances:
                    dialogue_context.append(self.dict.encode(msg['text']))


    def __getitem__(self, index):
        return {key: self.data[key][index] for key in self.data.keys()}

    def __len__(self):
        return len(self.data['target'])


class LandmarkDictionary(object):
    def __init__(self):
        self.i2landmark = ['Coffee Shop', 'Shop', 'Restaurant', 'Bank', 'Subway',
                           'Playfield', 'Theater', 'Bar', 'Hotel', 'Empty']
        self.landmark2i = {value: index for index, value in enumerate(self.i2landmark)}

    def encode(self, name):
        return self.landmark2i[name] + 1

    def decode(self, i):
        return self.i2landmark[i-1]

    def __len__(self):
        return len(self.i2landmark) + 1

class ActionAwareDictionary:

    def __init__(self):
        self.aware_id2act = ['ACTION:TURNLEFT', 'ACTION:TURNRIGHT', 'ACTION:FORWARD']
        self.aware_act2id = {v: k for k, v in enumerate(self.aware_id2act)}

    def encode(self, msg):
        if msg in self.aware_act2id:
            return self.aware_act2id[msg]+1
        return -1

    def decode(self, id):
        return self.aware_id2act[id-1]

    def __len__(self):
        return len(self.aware_id2act) + 1


class ActionAgnosticDictionary:
    def __init__(self):
        self.agnostic_id2act = ['LEFT', 'UP', 'RIGHT', 'DOWN', 'STAYED']
        self.agnostic_act2id = {v: k for k, v in enumerate(self.agnostic_id2act)}
        self.act_to_orientation = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    def encode(self, msg):
        return self.agnostic_act2id[msg] + 1

    def decode(self, id):
        return self.agnostic_id2act[id-1]

    def encode_from_location(self, old_loc, new_loc):
        """Determine if tourist went up, down, left, or right"""
        step_to_dir = {
            0: {
                1: 'UP',
                -1: 'DOWN',
                0: 'STAYED'
            },
            1: {
                0: 'LEFT',
            },
            -1: {
                0: 'RIGHT'
            }
        }

        step = [new_loc[0] - old_loc[0], new_loc[1] - old_loc[1]]
        act = self.agnostic_act2id[step_to_dir[step[0]][step[1]]]
        return act + 1

    def __len__(self):
        return len(self.agnostic_id2act) + 1


class Map(object):
    def __init__(self, data_dir, neighborhoods, include_empty_corners=True):
        super(Map, self).__init__()
        self.coord_to_landmarks = dict()
        self.include_empty_corners = include_empty_corners
        self.landmark_dict = LandmarkDictionary()
        self.data_dir = data_dir
        self.landmarks = dict()

        self.boundaries = dict()
        self.boundaries['hellskitchen'] = [3, 3]
        self.boundaries['williamsburg'] = [2, 8]
        self.boundaries['eastvillage'] = [3, 4]
        self.boundaries['fidi'] = [2, 3]
        self.boundaries['uppereast'] = [3, 3]

        for neighborhood in neighborhoods:
            self.coord_to_landmarks[neighborhood] = [[[] for _ in range(self.boundaries[neighborhood][1] * 2 + 4)]
                                                     for _ in range(self.boundaries[neighborhood][0] * 2 + 4)]
            self.landmarks[neighborhood] = json.load(open(os.path.join(data_dir, "{}_map.json".format(neighborhood))))
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
            if boundaries[0] >= landmark['x'] * 2 <= boundaries[2] and boundaries[1] >= landmark['y'] <= boundaries[3]:
                landmark_list.append(landmark)
        return landmark_list


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


def load_features(data_dir, neighborhoods):
    textfeatures = dict()
    for neighborhood in neighborhoods:
        textfeatures[neighborhood] = json.load(open(os.path.join(data_dir, 'text_%s.json' % neighborhood)))
    return textfeatures


def create_obs_dict(textfeatures, neighborhoods):
    obs_vocab = set([])
    for neighborhood in neighborhoods:
        # build textrecognition vocab
        for k, obs in textfeatures[neighborhood].items():
            obs_vocab |= set([o['lex_recog'] for o in obs])

    obs_i2s = list(obs_vocab)
    obs_s2i = {k: i for i, k in enumerate(obs_i2s)}

    return obs_i2s, obs_s2i


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
    def __init__(self, textfeatures, s2i):
        self.textfeatures = textfeatures
        self.s2i = s2i

    def get(self, neighborhood, loc):
        obs = list()
        for key in get_orientation_keys(loc[0], loc[1]):
            obs.extend([self.s2i[o['lex_recog']] for o in self.textfeatures[neighborhood][key]])
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
                obs.extend([self.X[self.k2i[o['lex_recog']]] for o in self.textfeatures[neighborhood][key]])
            else:
                obs.extend([self.f.get_word_vector(o['lex_recog']) for o in self.textfeatures[neighborhood][key]])

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


def step_agnostic(action, loc, boundaries):
    """Return new location after """
    new_loc = copy.deepcopy(loc)
    step = {'UP': (0, 1), 'RIGHT': (1, 0), 'DOWN': (0, -1), 'LEFT': (-1, 0)}[action]
    new_loc[0] = min(max(loc[0] + step[0], boundaries[0]), boundaries[2])
    new_loc[1] = min(max(loc[1] + step[1], boundaries[1]), boundaries[3])
    return new_loc


def step_aware(action, loc, boundaries):
    orientations = ['N', 'E', 'S', 'W']
    steps = dict()
    steps['N'] = [0, 1]
    steps['E'] = [1, 0]
    steps['S'] = [0, -1]
    steps['W'] = [-1, 0]

    new_loc = copy.deepcopy(loc)
    if action == 'ACTION:TURNLEFT':
        # turn left
        new_loc[2] = (new_loc[2] - 1) % 4

    if action == 'ACTION:TURNRIGHT':
        # turn right
        new_loc[2] = (new_loc[2] + 1) % 4

    if action == 'ACTION:FORWARD':
        # move forward
        orientation = orientations[loc[2]]
        new_loc[0] = new_loc[0] + steps[orientation][0]
        new_loc[1] = new_loc[1] + steps[orientation][1]

        new_loc[0] = min(max(new_loc[0], boundaries[0]), boundaries[2])
        new_loc[1] = min(max(new_loc[1], boundaries[1]), boundaries[3])
    return new_loc


def get_collate_fn(cuda=True):
    def _collate_fn(data):
        batch = dict()
        for k in data[0].keys():
            k_data = [data[i][k] for i in range(len(data))]
            if k in ['goldstandard', 'fasttext', 'textrecog', 'landmarks']:
                batch[k], _ = list_to_tensor(k_data)
            if k in ['observations', 'actions']:
                batch[k], batch[k+'_mask'] = list_to_tensor(k_data)
            if k  == 'utterance':
                batch['utterance'], batch['utterance_mask'] = list_to_tensor(k_data)
            if k in ['target']:
                batch[k] = torch.LongTensor(k_data)
        return to_variable(batch, cuda=cuda)
    return _collate_fn

def get_max_dimensions(arr):
    """Recursive function to calculate max dimensions of
       tensor (given a multi-dimensional list of arbitrary depth)
    """
    if not isinstance(arr, list):
        return []

    if len(arr) == 0:
        return [0]

    dims = None
    for a in arr:
        if dims is None:
            dims = get_max_dimensions(a)
        else:
            dims = [max(x, y) for x, y in zip(dims, get_max_dimensions(a))]
    return [len(arr)] + dims


def fill(ind, data_arr, value_tensor, mask_tensor):
    """Recursively fill tensor with values from multidimensional array
    """
    if not isinstance(data_arr, list):
        value_tensor[tuple(ind)] = data_arr
        mask_tensor[tuple(ind)] = 1.0
    else:
        for i, a in enumerate(data_arr):
            fill(ind + [i], a, value_tensor, mask_tensor)

def list_to_tensor(arr, pad_value=0):
    """Convert multi-dimensional array into tensor. Also returns mask.
    """
    dims = get_max_dimensions(arr)
    val_tensor = torch.LongTensor(*dims).fill_(pad_value)
    mask_tensor = torch.FloatTensor(*dims).zero_()
    fill([], arr, val_tensor, mask_tensor)
    return val_tensor, mask_tensor

def to_variable(obj, cuda=True):
    if torch.is_tensor(obj):
        var = Variable(obj)
        if cuda:
            var = var.cuda()
        return var
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [to_variable(x, cuda=cuda) for x in obj]
    if isinstance(obj, dict):
        return {k: to_variable(v, cuda=cuda) for k, v in obj.items()}
