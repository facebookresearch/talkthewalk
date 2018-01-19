import os
import json
import torch
from torch.autograd import Variable

data_dir = os.environ.get('TALKTHEWALK_DATADIR', './data')

class Landmarks(object):

    def __init__(self, neighborhoods, include_empty_corners=False):
        super(Landmarks, self).__init__()
        self.coord_to_idx, self.idx_to_coord, self.landmarks, self.types = {}, {}, {}, set([])

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


# get right orientation st you're facing landmarks
def get_orientation_keys(x, y, cross_the_street=False):
    if x % 2 == 0 and y % 2 == 0:
        k = ["{}_{}_{}".format(x, y, 'W1'),
             "{}_{}_{}".format(x, y, 'W2'),
             "{}_{}_{}".format(x, y, 'S1'),
             "{}_{}_{}".format(x, y, 'S1')]
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

    def __init__(self, file):
        import fastText

    def get(self, neighborhood, x, y):
        pass


class ResnetFeatures:
    def __init__(self, file):
        self.resnetfeatures = json.load(open(file))

    def get(self, neighborhood, x, y):
        pass

def load_data(configurations, feature_loader, landmark_map):
    X_data, landmark_data, y_data = list(), list(), list()

    for config in configurations:
        neighborhood = config['neighborhood']
        x, y = config['target_location'][:2]
        min_x, min_y = config['boundaries'][:2]

        obs = feature_loader.get(neighborhood, x, y)

        if landmark_map.has_landmarks(neighborhood, x, y):
            X_data.append(obs)
            landmarks, label_index = landmark_map.get_landmarks(neighborhood, min_x, min_y, x, y)
            landmark_data.append(landmarks)
            y_data.append(label_index)

    return X_data, landmark_data, y_data


def create_batch(b, landmarks, t, cuda=False):
    max_len, bsz = max(len(s) for s in b), len(b)
    batch = torch.LongTensor(max_len, bsz).zero_()
    for ii in range(bsz):
        for jj in range(len(b[ii])):
            batch[jj][ii] = b[ii][jj]

    landmark_lens = [len(l) for l in landmarks]
    max_landmarks = max(landmark_lens)
    max_landmarks_per_coord = max([max([len(x) for x in l]) for l in landmarks])

    l_batch = torch.LongTensor(bsz, max_landmarks, max_landmarks_per_coord).zero_()
    mask = torch.FloatTensor(bsz, max_landmarks).fill_(0.0)

    for i, ls in enumerate(landmarks):
        for j, l in enumerate(ls):
            l_batch[i, j, :len(l)] = torch.LongTensor(l)
        mask[i, :len(ls)] = 1.0

    batch, l_batch, mask, t = Variable(batch), Variable(l_batch), Variable(mask), Variable(torch.LongTensor(t)).unsqueeze(-1)
    if cuda:
        batch, l_batch, mask, t = batch.cuda(), l_batch.cuda(), mask.cuda(), t.cuda()

    return batch, l_batch, mask, t