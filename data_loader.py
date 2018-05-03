import copy
import os
import itertools
import json
import numpy
import random
import torch
from torch.autograd import Variable
#
from sklearn.decomposition import PCA
data_dir = os.environ.get('TALKTHEWALK_DATADIR', './data')

class Landmarks(object):

    def __init__(self, neighborhoods, include_empty_corners=False):
        super(Landmarks, self).__init__()
        self.coord_to_idx, self.idx_to_coord, self.landmarks, self.types = {}, {}, {}, set([])

        self.global_coord_to_idx = dict()
        self.idx_to_global_coord = list()

        self.boundaries = dict()
        self.boundaries['hellskitchen'] = [3, 3]
        self.boundaries['williamsburg'] = [2, 8]
        self.boundaries['eastvillage'] = [3, 4]
        self.boundaries['fidi'] = [2, 3]
        self.boundaries['uppereast'] = [3, 3]

        for neighborhood in neighborhoods:
            data = json.load(open(os.path.join(data_dir, "{}_map.json".format(neighborhood))))
            self.landmarks[neighborhood] = {}
            self.coord_to_idx[neighborhood] = {}
            self.idx_to_coord[neighborhood] = []

            x_offset = {"NW": 0, "SW": 0, "NE": 1, "SE": 1}
            y_offset = {"NW": 1, "SW": 0, "NE": 1, "SE": 0}

            for d in data:
                coord = (d['x']*2 + x_offset[d['orientation']], d['y']*2 + y_offset[d['orientation']])
                if coord not in self.coord_to_idx[neighborhood]:
                    self.coord_to_idx[neighborhood][coord] = len(self.idx_to_coord[neighborhood])
                    self.idx_to_coord[neighborhood].append(coord)
                    self.landmarks[neighborhood][coord] = []
                self.landmarks[neighborhood][coord].append(d['type'])
                self.types.add(d['type'])

            if include_empty_corners:
                self.types.add("Empty")
                for i in range(self.boundaries[neighborhood][0]*2 + 4):
                    for j in range(self.boundaries[neighborhood][1] * 2 + 4):
                        self.global_coord_to_idx[(neighborhood, i, j)]= len(self.idx_to_global_coord)
                        self.idx_to_global_coord.append((neighborhood, i, j))
                        coord = (i, j)
                        if coord not in self.landmarks[neighborhood]:
                            self.landmarks[neighborhood][coord] = ["Empty"]
                            self.coord_to_idx[neighborhood][coord] = len(self.idx_to_coord[neighborhood])
                            self.idx_to_coord[neighborhood].append(coord)

        self.itos = list(self.types)
        self.stoi = {k:i for i, k in enumerate(self.itos)}

    def has_landmarks(self, neighborhood, x, y):
        return (x, y) in self.coord_to_idx[neighborhood]

    def get_landmarks(self, neighborhood, min_x, min_y, x, y):
        assert (x, y) in self.coord_to_idx[neighborhood], "x, y coordinates do not have landmarks"
        landmarks = list()
        label_index = None
        k = 0
        for coord in self.idx_to_coord[neighborhood]:
            if x == coord[0] and y == coord[1]:
                label_index = k
                landmarks.append([self.stoi[l_type] + 1 for l_type in self.landmarks[neighborhood][coord]])
                k += 1
            else:
                if min_x <= coord[0] < min_x + 4 and min_y <= coord[1] < min_y + 4:
                    # if all([t not in self.landmarks[neighborhood][(x, y)] for t in self.landmarks[neighborhood][coord]]):
                    if True:
                        landmarks.append([self.stoi[l_type] + 1 for l_type in self.landmarks[neighborhood][coord]])
                        k += 1

        assert label_index is not None
        assert len(landmarks) == 16, "{}_{}_{}".format(x, y, neighborhood)

        return landmarks, label_index

    def get_landmarks_2d(self, neighborhood, boundaries, target_loc):
        assert tuple(target_loc[:2]) in self.coord_to_idx[neighborhood], "x, y coordinates do not have landmarks"
        landmarks = [[[] for _ in range(4)] for _ in range(4)]
        label_index = None
        k = 0
        for coord in self.idx_to_coord[neighborhood]:
            normalized_coord = coord[0]-boundaries[0], coord[1]-boundaries[1]
            if target_loc[0] == coord[0] and target_loc[1] == coord[1]:
                label_index = normalized_coord
                landmarks[normalized_coord[0]][normalized_coord[1]].extend([self.stoi[l_type] + 1 for l_type in self.landmarks[neighborhood][coord]])
                k += 1
            else:
                if boundaries[0] <= coord[0] <= boundaries[2] and boundaries[1] <= coord[1] <= boundaries[3]:
                    landmarks[normalized_coord[0]][normalized_coord[1]].extend([self.stoi[l_type] + 1 for l_type in self.landmarks[neighborhood][coord]])
                    k += 1

        assert label_index is not None

        return landmarks, label_index

    def get_softmax_idx(self, neighborhood, min_x, min_y, x, y):
        assert (x, y) in self.coord_to_idx[neighborhood], "x, y coordinates do not have landmarks"
        landmarks = list()
        label_index = None
        k = 0
        for coord in self.idx_to_coord[neighborhood]:
            if x == coord[0] and y == coord[1]:
                label_index = k
                landmarks.append([self.global_coord_to_idx[(neighborhood, coord[0], coord[1])]])
                k += 1
            else:
                if min_x <= coord[0] < min_x + 4 and min_y <= coord[1] < min_y + 4:
                    # if all([t not in self.landmarks[neighborhood][(x, y)] for t in self.landmarks[neighborhood][coord]]):
                    if True:
                        landmarks.append([self.global_coord_to_idx[(neighborhood, coord[0], coord[1])]])
                        k += 1

        assert label_index is not None
        assert len(landmarks) == 16, "{}_{}_{}".format(x, y, neighborhood)

        return landmarks, label_index

# get right orientation st you're facing landmarks
def get_orientation_keys(x, y, cross_the_street=False):
    if x % 2 == 0 and y % 2 == 0:
        k = ["{}_{}_{}".format(x, y, 'W1'),
             "{}_{}_{}".format(x, y, 'W2'),
             "{}_{}_{}".format(x, y, 'S1'),
             "{}_{}_{}".format(x, y, 'S2')]
        if cross_the_street:
            k += ["{}_{}_{}".format(x+1, y, 'W1'),
                  "{}_{}_{}".format(x+1, y, 'W2'),
                  "{}_{}_{}".format(x, y+1, 'S1'),
                  "{}_{}_{}".format(x, y+1, 'S2')]
        return k
    if x % 2 == 1 and y % 2 == 0:
        k = ["{}_{}_{}".format(x, y, 'E1'),
             "{}_{}_{}".format(x, y, 'E2'),
             "{}_{}_{}".format(x, y, 'S1'),
             "{}_{}_{}".format(x, y, 'S2')]
        if cross_the_street:
            k += ["{}_{}_{}".format(x-1, y, 'E1'),
                  "{}_{}_{}".format(x-1, y, 'E2'),
                  "{}_{}_{}".format(x, y+1, 'S1'),
                  "{}_{}_{}".format(x, y+1, 'S2')]
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

def load_features(neighborhoods):
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

    def __init__(self, landmark_map, orientation_aware=False):
        self.landmark_map = landmark_map
        self.allowed_orientations = {'NW': [3, 0], 'SW': [2, 3], 'NE': [0, 1], 'SE': [1, 2]}
        self.mod2orientation = {(0, 0): 'SW', (1, 0): 'SE', (0, 1): 'NW', (1, 1): 'NE'}
        self.orientation_aware = orientation_aware

    def get(self, neighborhood, loc):
        if self.orientation_aware:
            mod = (loc[0]%2, loc[1]%2)
            orientation = self.mod2orientation[mod]
            if loc[2] in self.allowed_orientations[orientation]:
                return [self.landmark_map.stoi[t] + 1 for t in
                        self.landmark_map.landmarks[neighborhood][tuple(loc[:2])]]
            else:
                return [11]
        else:
            return [self.landmark_map.stoi[t] + 1 for t in self.landmark_map.landmarks[neighborhood][tuple(loc[:2])]]


class TextrecogFeatures:

    def __init__(self, textfeatures, s2i):
        self.textfeatures = textfeatures
        self.s2i = s2i

    def get(self, neighborhood, x, y):
        obs = list()
        for key in get_orientation_keys(x, y):
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


    def get(self, neighborhood, x, y):
        obs = list()
        for key in get_orientation_keys(x, y):
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

    def get(self, neighborhood, x, y):
        obs = list()
        for key in get_orientation_keys(x, y):
            if self.pca:
                obs.append(self.X[self.k2i["{}_{}".format(neighborhood, key)]])
            else:
                obs.append(self.resnetfeatures[neighborhood][key])
        return obs


def step_agnostic(action, loc, boundaries):
    new_loc = copy.deepcopy(loc)
    step = [(0, 1), (1, 0), (0, -1), (-1, 0)][action]
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
    if action == 0:
        # turn left
        new_loc[2] = (new_loc[2] - 1) % 4

    if action  == 1:
        # turn right
        new_loc[2] = (new_loc[2] + 1) % 4

    if action == 2:
        # move forward
        orientation = orientations[loc[2]]
        new_loc[0] = new_loc[0] + steps[orientation][0]
        new_loc[1] = new_loc[1] + steps[orientation][1]

        new_loc[0] = min(max(new_loc[0], boundaries[0]), boundaries[2])
        new_loc[1] = min(max(new_loc[1], boundaries[1]), boundaries[3])
    return new_loc

def load_data(configurations, feature_loaders, landmark_map, softmax='location', num_steps=2, samples_per_configuration=None):
    X_data, action_data, landmark_data, y_data = {k: list() for k in feature_loaders.keys()}, list(), list(), list()

    action_set = [[0, 1, 2, 3]]*(num_steps-1)
    all_possible_actions = list(itertools.product(*action_set))

    # all_possible_actions = [[random.randint(0, 3) for _ in range(num_steps-1)]]

    for config in configurations:
        for a in all_possible_actions:
            neighborhood = config['neighborhood']
            target_loc = config['target_location']
            boundaries = config['boundaries']

            obs = {k: list() for k in feature_loaders.keys()}
            actions = list()
            loc = copy.deepcopy(config['target_location'])
            for p in range(num_steps):
                for k, feature_loader in feature_loaders.items():
                    obs[k].append(feature_loader.get(neighborhood, loc))

                if p != num_steps - 1:
                    sampled_act = a[p]
                    actions.append(sampled_act)
                    loc = step_agnostic(sampled_act, loc, boundaries)

            if num_steps == 1:
                actions.append(0)

            for k in feature_loaders.keys():
                X_data[k].append(obs[k])

            action_data.append(actions)
            if softmax == 'location':
                landmarks, label_index = landmark_map.get_softmax_idx(neighborhood, boundaries, target_loc)
            else:
                landmarks, label_index = landmark_map.get_landmarks_2d(neighborhood, boundaries, target_loc)
            landmark_data.append(landmarks)
            y_data.append(label_index)

    return X_data, action_data, landmark_data, y_data


def create_batch(X, landmarks, y, cuda=False):
    bsz = len(y)
    batch = dict()
    if 'resnet' in X:
        batch['resnet'] = torch.FloatTensor(X['resnet'])
    if 'fasttext' in X:
        max_len = max(len(s) for s in X['fasttext'])
        batch['fasttext'] = torch.FloatTensor(bsz, max_len, 100).zero_()
        for ii in range(bsz):
            for jj in range(len(X['fasttext'][ii])):
                batch['fasttext'][ii, jj, :] = torch.from_numpy(X['fasttext'][ii][jj])
    if 'textrecog' in X:
        max_len = max(len(s) for s in X['textrecog'])
        batch['textrecog'] = torch.LongTensor(bsz, max_len).zero_()
        for ii in range(bsz):
            for jj in range(len(X['textrecog'][ii])):
                batch['textrecog'][ii, jj] = X['textrecog'][ii][jj]
    if 'goldstandard' in X:
        max_len = max(len(s) for s in X['goldstandard'])
        batch['goldstandard'] = torch.LongTensor(bsz, max_len).zero_()
        for ii in range(bsz):
            for jj in range(len(X['goldstandard'][ii])):
                batch['goldstandard'][ii, jj] = X['goldstandard'][ii][jj]


    landmark_lens = [len(l) for l in landmarks]
    max_landmarks = max(landmark_lens)
    max_landmarks_per_coord = max([max([len(x) for x in l]) for l in landmarks])

    landmark_batch = torch.LongTensor(bsz, max_landmarks, max_landmarks_per_coord).zero_()
    mask = torch.FloatTensor(bsz, max_landmarks).fill_(0.0)

    for i, ls in enumerate(landmarks):
        for j, l in enumerate(ls):
            landmark_batch[i, j, :len(l)] = torch.LongTensor(l)
        mask[i, :len(ls)] = 1.0

    return to_variable((batch, landmark_batch, mask, torch.LongTensor(y).unsqueeze(-1)), cuda=cuda)


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
