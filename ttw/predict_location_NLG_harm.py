import argparse
import os
import json
import torch
import torch.optim as optim

from torch.autograd import Variable

from talkthewalk.predict_location_language import Guide
from talkthewalk.train_tourist_supervised import Tourist, load_data, show_samples
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

    return to_variable(
        [obs_tensor, obs_seqlen_tensor, act_tensor, act_seqlen_tensor, landmark_tensor, torch.LongTensor(tgts)],
        cuda=cuda)


def cache(data, tourist, batch_sz, max_sample_length=15, cuda=True, decoding_strategy='beam_search'):
    print('Caching tourist messages, decoding = ' + decoding_strategy)
    messages = torch.LongTensor(len(data[0]), max_sample_length)
    mask = torch.FloatTensor(len(data[0]), max_sample_length)

    observations = data[0]
    actions = data[1]
    landmarks = data[3]
    targets = data[4]

    for jj in range(0, len(observations), batch_sz):
        batch = create_batch(observations[jj:jj + batch_sz], actions[jj:jj + batch_sz], landmarks[jj:jj + batch_sz],
                             targets[jj:jj + batch_sz], cuda=cuda)
        obs_batch, obs_seq_len_batch, act_batch, act_seq_len_batch, landmark_batch, tgt_batch = batch

        out = tourist.forward(obs_batch, obs_seq_len_batch, act_batch, act_seq_len_batch,
                              decoding_strategy=decoding_strategy, train=False,
                              max_sample_length=max_sample_length)

        messages[jj:jj + batch_sz, :] = out['preds'].cpu().data
        mask[jj:jj + batch_sz, :] = out['mask'].cpu().data

    return messages, mask, None, landmarks, targets


def create_batch_guide(landmarks, tgts, cuda=True):
    batch_sz = len(landmarks)
    max_landmarks_per_coord = max([max([max([len(y) for y in x]) for x in l]) for l in landmarks])
    landmark_tensor = torch.LongTensor(batch_sz, 4, 4, max_landmarks_per_coord).zero_()

    for i, ls in enumerate(landmarks):
        for j in range(4):
            for k in range(4):
                landmark_tensor[i, j, k, :len(landmarks[i][j][k])] = torch.LongTensor(landmarks[i][j][k])

    return to_variable([landmark_tensor, torch.LongTensor(tgts)], cuda=cuda)


def epoch(data, tourist, guide, g_opt=None, t_opt=None,
          batch_sz=256, cuda=True, cached=True,
          decoding_strategy='beam_search'):
    if not cached:
        observations = data[0]
        actions = data[1]

    landmarks = data[3]
    targets = data[4]

    accuracy, total = 0.0, 0.0

    for jj in range(0, len(landmarks), batch_sz):
        if not cached:
            batch = create_batch(observations[jj:jj + batch_sz], actions[jj:jj + batch_sz],
                                 landmarks[jj:jj + batch_sz], targets[jj:jj + batch_sz], cuda=cuda)
            obs_batch, obs_seq_len_batch, act_batch, act_seq_len_batch, landmark_batch, tgt_batch = batch

            out = tourist.forward(obs_batch, obs_seq_len_batch, act_batch, act_seq_len_batch,
                                  decoding_strategy=decoding_strategy, train=False)
            M = out['preds']
            mask = out['mask']
        else:
            M = Variable(data[0][jj:jj + batch_sz]).cuda()
            mask = Variable(data[1][jj:jj + batch_sz]).cuda()
            landmark_batch, tgt_batch = create_batch_guide(landmarks[jj:jj + batch_sz], targets[jj:jj + batch_sz])

        loss, acc = guide.forward(M, mask, landmark_batch, tgt_batch)

        reward = -loss.squeeze()
        loss = loss.sum()

        total += batch_sz
        accuracy += acc * batch_sz

        if g_opt is not None:
            g_opt.zero_grad()
            loss.backward()
            g_opt.step()

        if t_opt is not None:
            # reinforce
            probs = out['probs']
            mask = out['mask']
            sampled_ind = out['preds']

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

    return accuracy / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--cache', action='store_true', help="Cache samples from tourist model and train on that")
    parser.add_argument('--tourist-model', type=str)
    parser.add_argument('--guide-model', type=str)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-sz', type=int, default=512)
    parser.add_argument('--train-guide', action='store_true')
    parser.add_argument('--train-tourist', action='store_true')
    parser.add_argument('--exp-name', default='test')
    parser.add_argument('--decoding-strategy', type=str)

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
    if args.guide_model is not None:
        guide = Guide.load(args.guide_model)
    else:
        guide = Guide(128, 256, len(text_dict), apply_masc=True, T=3)

    if args.cuda:
        tourist = tourist.cuda()
        guide = guide.cuda()

    if args.train_guide:
        logger.info('Train guide (supervised)')
        g_opt = optim.Adam(guide.parameters())

    if args.train_tourist:
        logger.info('Train tourist (supervised)')
        t_opt = optim.Adam(tourist.parameters())

    show_samples(train_data, tourist, text_dict, landmark_map, cuda=args.cuda, decoding_strategy=args.decoding_strategy)

    if args.cache:
        train_data = cache(train_data, tourist, args.batch_sz, cuda=args.cuda, decoding_strategy=args.decoding_strategy)
        valid_data = cache(valid_data, tourist, args.batch_sz, cuda=args.cuda, decoding_strategy=args.decoding_strategy)
        test_data = cache(test_data, tourist, args.batch_sz, cuda=args.cuda, decoding_strategy=args.decoding_strategy)

    best_train_acc, best_valid_acc, best_test_acc = 0.0, 0.0, 0.0
    for i in range(args.num_epochs):
        g_optim = None
        if args.train_guide:
            g_optim = g_opt

        t_optim = None
        if args.train_tourist:
            t_optim = t_opt

        train_acc = epoch(train_data, tourist, guide, cuda=args.cuda, g_opt=g_optim, t_opt=t_optim,
                          decoding_strategy=args.decoding_strategy, cached=args.cache)
        valid_acc = epoch(valid_data, tourist, guide, cuda=args.cuda, decoding_strategy=args.decoding_strategy,
                          cached=args.cache)
        test_acc = epoch(test_data, tourist, guide, cuda=args.cuda, decoding_strategy=args.decoding_strategy,
                         cached=args.cache)

        logger.info(
            'Epoch: {} -- Train acc: {}, Valid acc: {}, Test acc: {}'.format(i + 1, train_acc * 100, valid_acc * 100,
                                                                             test_acc * 100))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            tourist.save(os.path.join(exp_dir, 'tourist.pt'))
            guide.save(os.path.join(exp_dir, 'guide.pt'))
