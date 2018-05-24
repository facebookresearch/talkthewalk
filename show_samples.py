import argparse
import os
import json
import random
import torch
import torch.optim as optim

from torch.autograd import Variable
from sklearn.utils import shuffle


from talkthewalk.predict_location_language import Guide
from talkthewalk.train_tourist_supervised import Tourist, load_data, show_samples
from talkthewalk.dict import PAD_TOKEN, END_TOKEN
from talkthewalk.utils import create_logger
from talkthewalk.data_loader import Landmarks, GoldstandardFeatures, neighborhoods, to_variable
from talkthewalk.dict import Dictionary

data_dir = os.environ.get('TALKTHEWALK_DATADIR', './data')

train_configs = json.load(open(os.path.join(data_dir, 'talkthewalk.train.json')))
valid_configs = json.load(open(os.path.join(data_dir, 'talkthewalk.valid.json')))
test_configs = json.load(open(os.path.join(data_dir, 'talkthewalk.test.json')))

text_dict = Dictionary(file=os.path.join(data_dir, 'dict.txt'), min_freq=3)
landmark_map = Landmarks(neighborhoods, include_empty_corners=True)
loader = GoldstandardFeatures(landmark_map)

train_data = load_data(train_configs, text_dict, loader, landmark_map)
valid_data = load_data(valid_configs, text_dict, loader, landmark_map)
test_data = load_data(test_configs, text_dict, loader, landmark_map)

tourist_sl = Tourist.load('/u/devries/exp/tourist_sl_2/tourist.pt').cuda()
tourist_rl = Tourist.load('/u/devries/exp/tourist_rl_2/tourist.pt').cuda()

indices = []
for _ in range(5):
    indices.append(random.randint(0, len(test_data[0]) -1))
print('supervised, greedy')
show_samples(test_data, tourist_sl, text_dict, landmark_map, indices=indices, decoding_strategy='greedy')
print(); print()
print('supervised, beam')
show_samples(test_data, tourist_sl, text_dict, landmark_map, indices=indices, decoding_strategy='beam_search')
print(); print()
print('supervised, sample')
show_samples(test_data, tourist_sl, text_dict, landmark_map, indices=indices, decoding_strategy='sample')
print(); print()
print('policy grad, greedy')
show_samples(test_data, tourist_rl, text_dict, landmark_map, indices=indices, decoding_strategy='greedy')
print(); print()
print('policy grad, sample')
show_samples(test_data, tourist_rl, text_dict, landmark_map, indices=indices, decoding_strategy='sample')