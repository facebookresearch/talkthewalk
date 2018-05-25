import random

from ttw.data_loader import TalkTheWalkLanguage
from ttw.train.train_tourist import TouristLanguage, show_samples

train_data = TalkTheWalkLanguage('./data', 'train')

tourist_sl = TouristLanguage.load('/u/devries/Documents/talkthewalk/exp/tourist_sl4/tourist.pt').cuda()
# tourist_rl = TouristLanguage.load('/u/devries/exp/tourist_rl_2/tourist.pt').cuda()

indices = []
for _ in range(5):
    indices.append(random.randint(0, len(train_data) -1))

indices = [32, 245, 560, 750, 1200, 2467, 2500]
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