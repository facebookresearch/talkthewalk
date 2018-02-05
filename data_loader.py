import copy
import os
import itertools
import json
import numpy
import random
import torch
from torch.autograd import Variable

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

    def get_landmarks_2d(self, neighborhood, min_x, min_y, x, y):
        assert (x, y) in self.coord_to_idx[neighborhood], "x, y coordinates do not have landmarks"
        landmarks = [[[] for _ in range(4)] for _ in range(4)]
        label_index = None
        k = 0
        for coord in self.idx_to_coord[neighborhood]:
            normalized_coord = coord[0]-min_x, coord[1]-min_y
            if x == coord[0] and y == coord[1]:
                label_index = normalized_coord
                landmarks[normalized_coord[0]][normalized_coord[1]].extend([self.stoi[l_type] + 1 for l_type in self.landmarks[neighborhood][coord]])
                k += 1
            else:
                if min_x <= coord[0] < min_x + 4 and min_y <= coord[1] < min_y + 4:
                    # if all([t not in self.landmarks[neighborhood][(x, y)] for t in self.landmarks[neighborhood][coord]]):
                    if True:
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

    def __init__(self, landmark_map):
        self.landmark_map = landmark_map

    def get(self, neighborhood, x, y):
        obs = list()
        if (x, y) in self.landmark_map.landmarks[neighborhood]:
            obs.extend([self.landmark_map.stoi[t] + 1 for t in self.landmark_map.landmarks[neighborhood][(x, y)]])
        return obs

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


def load_data_multiple_step(configurations, feature_loaders, landmark_map, softmax='location', num_steps=2, samples_per_configuration=None):
    X_data, action_data, landmark_data, y_data = {k: list() for k in feature_loaders.keys()}, list(), list(), list()

    action_set = [[0, 1, 2, 3]]*(num_steps-1)
    all_possible_actions = list(itertools.product(*action_set))

    # all_possible_actions = [[random.randint(0, 3) for _ in range(num_steps-1)]]


    for config in configurations:
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
                    sampled_x = max(min(sampled_x + step[0], min_x+3), min_x)
                    sampled_y = max(min(sampled_y + step[1], min_y+3), min_y)

            if num_steps == 1:
                actions.append(0)
            # pred_upbound_act += prediction_upperbound(obs['goldstandard'], landmark_map, neighborhood, min_x, min_y, sampled_x, sampled_y,
            #                                       actions=actions)
            # pred_upbound += prediction_upperbound(obs['goldstandard'], landmark_map, neighborhood, min_x, min_y,
            #                                           sampled_x, sampled_y)

            if landmark_map.has_landmarks(neighborhood, x, y):
                for k in feature_loaders.keys():
                    X_data[k].append(obs[k])
                action_data.append(actions)
                if softmax == 'location':
                    landmarks, label_index = landmark_map.get_softmax_idx(neighborhood, min_x, min_y, x, y)
                else:
                    landmarks, label_index = landmark_map.get_landmarks_2d(neighborhood, min_x, min_y, x, y)
                landmark_data.append(landmarks)
                y_data.append(label_index)

    # print(pred_upbound/len(configurations), pred_upbound_act/len(configurations))

    return X_data, action_data, landmark_data, y_data


def load_data(configurations, feature_loaders, landmark_map, softmax='location'):
    X_data, landmark_data, y_data = {k: list() for k in feature_loaders.keys()}, list(), list()

    for config in configurations:
        neighborhood = config['neighborhood']
        x, y = config['target_location'][:2]
        min_x, min_y = config['boundaries'][:2]

        obs = {k: feature_loader.get(neighborhood, x, y) for k, feature_loader in feature_loaders.items()}

        if landmark_map.has_landmarks(neighborhood, x, y):
            for k in feature_loaders.keys():
                X_data[k].append(obs[k])
            if softmax == 'location':
                landmarks, label_index = landmark_map.get_softmax_idx(neighborhood, min_x, min_y, x, y)
            else:
                landmarks, label_index = landmark_map.get_landmarks(neighborhood, min_x, min_y, x, y)
            landmark_data.append(landmarks)
            y_data.append(label_index)

    return X_data, landmark_data, y_data


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


# TODO function to create torch tensor from list of lists
def pad(tensor, type=torch.FloatTensor):
    raise NotImplementedError()


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
