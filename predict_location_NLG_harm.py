import argparse
import os
import json
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

def create_batch(observations, actions, landmarks, tgts, cuda=True):
    batch_sz = len(observations)
    obs_seq_len = [len(s) for s in observations]
    obs_seqlen_tensor = torch.LongTensor(obs_seq_len)
    max_steps = max(obs_seq_len)
    max_len = max([max([len(x) for x in l]) for l in observations])
    obs_tensor = torch.LongTensor(batch_sz, max_steps, max_len).zero_()
    for ii in range(batch_sz):
        for jj in range(len(observations[ii])):
            for kk in range(len(observations[ii][jj])):
                obs_tensor[ii, jj, kk] = observations[ii][jj][kk]

    act_seqlen_tensor = torch.LongTensor([len(s) for s in actions])
    max_len = max([len(s) for s in actions])
    act_tensor = torch.LongTensor(batch_sz, max_len).zero_()
    for ii in range(batch_sz):
        for jj in range(len(actions[ii])):
            act_tensor[ii][jj] = actions[ii][jj]

    max_landmarks_per_coord = max([max([max([len(y) for y in x]) for x in l]) for l in landmarks])
    landmark_tensor = torch.LongTensor(batch_sz, 4, 4, max_landmarks_per_coord).zero_()

    for i, ls in enumerate(landmarks):
        for j in range(4):
            for k in range(4):
                landmark_tensor[i, j, k, :len(landmarks[i][j][k])] = torch.LongTensor(landmarks[i][j][k])


    return to_variable([obs_tensor, obs_seqlen_tensor, act_tensor, act_seqlen_tensor, landmark_tensor, torch.LongTensor(tgts)], cuda=cuda)


def epoch(data, tourist, guide, g_opt=None, t_opt=None, batch_sz=256, cuda=True):
    observations = data[0]
    actions = data[1]
    landmarks = data[3]
    targets = data[4]

    accuracy, total = 0.0, 0.0

    for jj in range(0, len(observations), batch_sz):
        batch = create_batch(observations[jj:jj+batch_sz], actions[jj:jj+batch_sz], landmarks[jj:jj+batch_sz], targets[jj:jj+batch_sz], cuda=cuda)
        obs_batch, obs_seq_len_batch, act_batch, act_seq_len_batch, landmark_batch, tgt_batch = batch

        msg = tourist.forward(obs_batch, obs_seq_len_batch, act_batch, act_seq_len_batch, sample=True)

        loss, acc = guide.forward(msg['preds'], msg['mask'], landmark_batch, tgt_batch)

        reward = -loss.squeeze()
        loss = loss.sum()

        total += batch_sz
        accuracy += acc*batch_sz

        if g_opt is not None:
            g_opt.zero_grad()
            loss.backward()
            g_opt.step()

        if t_opt is not None:
            # reinforce
            probs = msg['probs']
            mask = msg['mask']
            sampled_ind = msg['preds']

            loss = 0.0

            for k in range(probs.size(1)):
                prob = probs[:, k, :]
                ind = sampled_ind[:, k]
                selected_prob = torch.gather(prob, 1, ind.unsqueeze(-1))
                log_prob = torch.log(selected_prob + 1e-8).squeeze(-1)
                advantage = reward - reward.mean()
                loss -= (mask[:, k] * log_prob * advantage).sum()

            t_opt.zero_grad()
            loss.backward()
            t_opt.step()

    return accuracy/total



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--tourist-model', type=str)
    parser.add_argument('--guide-model', type=str)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-sz', type=int, default=256)
    parser.add_argument('--train-guide', action='store_true')
    parser.add_argument('--train-tourist', action='store_true')
    parser.add_argument('--exp-name', default='test')

    args = parser.parse_args()
    print(args)

    exp_dir = os.path.join(os.environ['TALKTHEWALK_EXPDIR'], args.exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    logger = create_logger(os.path.join(exp_dir, 'log.txt'))
    logger.info(args)

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

    tourist = Tourist.load(args.tourist_model)
    guide = Guide.load(args.guide_model)

    if args.cuda:
        tourist = tourist.cuda()
        guide = guide.cuda()

    if args.train_guide:
        logger.info('Train guide (supervised)')
        g_opt = optim.Adam(guide.parameters())

    if args.train_tourist:
        logger.info('Train tourist (supervised)')
        t_opt = optim.Adam(tourist.parameters())

    show_samples(train_data, tourist, text_dict, landmark_map, cuda=args.cuda)

    best_train_acc, best_valid_acc, best_test_acc = 0.0, 0.0, 0.0
    for i in range(args.num_epochs):
        g_optim = None
        if args.train_guide:
            g_optim = g_opt

        t_optim = None
        if args.train_tourist:
            t_optim = t_opt

        train_acc = epoch(train_data, tourist, guide, cuda=args.cuda, g_opt=g_optim, t_opt=t_optim)
        valid_acc = epoch(valid_data, tourist, guide, cuda=args.cuda)
        test_acc = epoch(test_data, tourist, guide, cuda=args.cuda)

        logger.info('Epoch: {} -- Train acc: {}, Valid acc: {}, Test acc: {}'.format(i+1, train_acc*100, valid_acc*100, test_acc*100))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            tourist.save(os.path.join(exp_dir, 'tourist.pt'))
            guide.save(os.path.join(exp_dir, 'guide.pt'))







