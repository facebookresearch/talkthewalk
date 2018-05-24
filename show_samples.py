import os
import json
import random


from ttw.train_tourist import TouristLanguage, show_samples
from ttw.data_loader import TalkTheWalkLanguage

train_data = TalkTheWalkLanguage('./data', 'train')

tourist_sl = TouristLanguage.load('/u/devries/Documents/talkthewalk/exp/tourist_sl/tourist.pt').cuda()
# tourist_rl = TouristLanguage.load('/u/devries/exp/tourist_rl_2/tourist.pt').cuda()

indices = []
for _ in range(5):
    indices.append(random.randint(0, len(train_data) -1))
print('supervised, greedy')
show_samples(train_data, tourist_sl, indices=indices, decoding_strategy='greedy')
print(); print()
print('supervised, beam')
show_samples(train_data, tourist_sl, indices=indices, decoding_strategy='beam_search')

# print(); print()
# print('supervised, sample')
# show_samples(test_data, tourist_sl, text_dict, landmark_map, indices=indices, decoding_strategy='sample')
# print(); print()
# print('policy grad, greedy')
# show_samples(test_data, tourist_rl, text_dict, landmark_map, indices=indices, decoding_strategy='greedy')
# print(); print()
# print('policy grad, sample')
# show_samples(test_data, tourist_rl, text_dict, landmark_map, indices=indices, decoding_strategy='sample')