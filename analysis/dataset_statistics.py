import json

f = './data/talkthewalk.train.json'
data = json.load(open(f))

worker_ids = dict()

k = 0.
tourist_forward = 0.0
tourist_actions = 0.0
tourist_utterance = 0.
utt_length = 0.
guide_utterance = 0.
length = 0.

for sample in data:
    if sample['tourist_worker_id'] not in worker_ids:
        worker_ids[sample['tourist_worker_id']] = True
    if sample['guide_worker_id'] not in worker_ids:
        worker_ids[sample['guide_worker_id']] = True

    for turn in sample['dialog']:
        if turn['id'] == 'Tourist':
            if 'ACTION:' in turn['text']:
                tourist_actions += 1
                if 'FORWARD' in turn['text']:
                    tourist_forward += 1
            else:
                tourist_utterance += 1
        else:
            guide_utterance += 1
    k+=1


print("Number of Turkers", len(worker_ids))
print("Average number of turns per dialogue", (tourist_actions+tourist_utterance+guide_utterance)/k)
print("Average number of actions per dialogue", (tourist_actions)/k)
print("Average number of tourist utterance per dialogue", (tourist_utterance)/k)
print("Average number of guide utterance per dialogue", (guide_utterance)/k)
print("Average number of forward per dialogue", (tourist_forward)/k)