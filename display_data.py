import json

f = './data/talkthewalk.train.json'
data = json.load(open(f))

print(type(data))

for sample in data:
    for turn in sample['dialog']:
        print(turn['id'], turn['text'])
    print('-'*80)