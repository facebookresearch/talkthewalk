"""Show all samples from the train set."""
import json
import argparse


f = './data/talkthewalk.train.json'
data = json.load(open(f))

for sample in data:
    tourist_action_list = list()
    for turn in sample['dialog']:
        if turn['id'] == 'Tourist' and 'ACTION' in turn['text']:
            tourist_action_list.append(turn['text'])
        else:
            if len(tourist_action_list) > 0:
                print("Tourist:  " + ' '.join(tourist_action_list))
                tourist_action_list = list()
            if turn['id'] == 'Tourist':
                print('Tourist:  ' + turn['text'])
            else:
                print('Guide:    ' + turn['text'])

    print('-'*80)