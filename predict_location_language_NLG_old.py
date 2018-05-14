import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from collections import deque

from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from data_loader import Landmarks, step_aware, to_variable
from utils import create_logger
from dict import Dictionary
from predict_location_multiple_step import MapEmbedding2d
from train_NLG import TrainLanguageGenerator
def str2bool(value):
    v = value.lower()
    if v in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def get_action(msg):
    msg_to_act = {'ACTION:TURNLEFT': 0, 'ACTION:TURNRIGHT': 1, 'ACTION:FORWARD': 2}
    return msg_to_act.get(msg, None)

def load_data(dataset, landmark_map, dictionary, full_dialogue=False, min_sent_length=2, num_past_utterances=5):
    Xs = list()
    landmarks = list()
    ys = list()
    for config in dataset:
        loc = config['start_location']
        boundaries = config['boundaries']

        dialogue = deque(maxlen=num_past_utterances)
        for msg in config['dialog']:
            if msg['id'] == 'Tourist':
                act = get_action(msg['text'])
                if act is not None:
                    loc = step_aware(act, loc, boundaries)
                elif len(msg['text'].split(' ')) > min_sent_length:
                    if full_dialogue:
                        dialogue.extend(dictionary.encode(msg['text']))
                        Xs.append(dialogue)
                    else:
                        Xs.append(dictionary.encode(msg['text']))
                    ls, y = landmark_map.get_landmarks_2d(config['neighborhood'], boundaries, loc)
                    landmarks.append(ls)
                    ys.append(y)
            elif full_dialogue:
                dialogue.extend(dictionary.encode(msg['text']))

    return Xs, landmarks, ys

def create_batch(Xs, landmarks, ys, use_cuda=False):
    batch_size = len(Xs)
    seq_lens = [len(seq) for seq in Xs]
    max_len = max(seq_lens)

    X_batch = torch.LongTensor(batch_size, max_len).zero_()
    mask = torch.FloatTensor(batch_size, max_len).zero_()
    for i, seq in enumerate(Xs):
        X_batch[i, :len(seq)] = torch.LongTensor(seq)
        mask[i, :len(seq)] = 1.0

    max_landmarks_per_coord = max([max([max([len(y) for y in x]) for x in l]) for l in landmarks])
    landmark_batch = torch.LongTensor(batch_size, 4, 4, max_landmarks_per_coord).zero_()

    for i, ls in enumerate(landmarks):
        for j in range(4):
            for k in range(4):
                landmark_batch[i, j, k, :len(landmarks[i][j][k])] = torch.LongTensor(landmarks[i][j][k])

    y_batch = torch.LongTensor(ys)

    return to_variable([X_batch, mask, landmark_batch, y_batch], cuda=use_cuda)


class LocationPredictor(nn.Module):

    def __init__(self, inp_emb_sz, hidden_sz, num_tokens, max_steps=1, condition_on_action=False, mask_conv=True, use_cuda=False):
        super(LocationPredictor, self).__init__()
        self.hidden_sz = hidden_sz
        self.use_cuda = use_cuda
        self.inp_emb_sz = inp_emb_sz
        self.num_tokens = num_tokens
        self.embedder = nn.Embedding(num_tokens, inp_emb_sz, padding_idx=0)
        self.encoder = nn.LSTM(inp_emb_sz, hidden_sz//2, batch_first=True, bidirectional=True)
        self.emb_map = MapEmbedding2d(11, hidden_sz)
        self.condition_on_action = condition_on_action
        self.mask_conv = mask_conv
        self.feat_control_emb = Parameter(torch.FloatTensor(hidden_sz).normal_(0.0, 0.1))
        self.feat_control_updater = nn.Linear(2*hidden_sz, hidden_sz)
        if condition_on_action:
            self.act_control_emb = Parameter(torch.FloatTensor(hidden_sz).normal_(0.0, 0.1))
            self.action_lin = nn.Linear(hidden_sz, 9)
            self.act_control_updater = nn.Linear(2*hidden_sz, hidden_sz)

        self.conv_weight = nn.Parameter(torch.FloatTensor(
            hidden_sz, hidden_sz, 3, 3))
        std = 1.0 / (hidden_sz * 9)
        self.conv_weight.data.uniform_(-std, std)
        self.max_steps = max_steps
        self.loss = nn.CrossEntropyLoss()

    def forward(self, Xs, seq_mask, landmarks, ys):
        batch_size = Xs.size(0)
        input_emb = self.embedder.forward(Xs)
        print(input_emb.size())
        hidden_states, _ = self.encoder.forward(input_emb)

        feature_msgs = list()
        controller = self.feat_control_emb.unsqueeze(0).repeat(batch_size, 1)
        for step in range(self.max_steps):
            score = torch.bmm(hidden_states, controller.unsqueeze(-1)).squeeze(-1)
            score = score - 1e30*(1.0-seq_mask)
            att_score = F.softmax(score, dim=-1)
            extracted_msg = torch.bmm(att_score.unsqueeze(1), hidden_states).squeeze()
            feature_msgs.append(extracted_msg)
            controller = self.feat_control_updater.forward(torch.cat([extracted_msg, controller], 1))

        feature_msg = torch.cat(feature_msgs, 1)

        l_emb = self.emb_map.forward(landmarks).permute(0, 3, 1, 2)
        l_embs = [l_emb]

        if self.condition_on_action:
            act_controller = self.act_control_emb.unsqueeze(0).repeat(batch_size, 1)
            for step in range(self.max_steps-1):
                score = torch.bmm(hidden_states, act_controller.unsqueeze(-1)).squeeze(-1)
                score = score - 1e30*(1-seq_mask)
                att_score = F.softmax(score, dim=-1)
                extracted_msg = torch.bmm(att_score.unsqueeze(1), hidden_states).squeeze()
                act_controller = self.act_control_updater.forward(torch.cat([extracted_msg, act_controller], 1))
                action_out = self.action_lin.forward(extracted_msg)
                if self.use_cuda:
                    out = Variable(torch.FloatTensor(batch_size, self.hidden_sz, 4, 4).zero_().cuda())
                else:
                    out = Variable(torch.FloatTensor(batch_size, self.hidden_sz, 4, 4).zero_())
                for i in range(batch_size):
                    selected_inp = l_embs[-1][i, :, :, :].unsqueeze(0)
                    mask = F.softmax(action_out[i], dim=0).resize(1, 1, 3, 3)
                    weight = mask * self.conv_weight
                    out[i, :, :, :] = F.conv2d(selected_inp, weight, padding=1).squeeze(0)
                l_embs.append(out)
        else:
            weight = self.conv_weight
            if self.mask_conv:
                if self.use_cuda:
                    mask = torch.FloatTensor(1, 1, 3, 3).cuda().zero_()
                else:
                    mask = torch.FloatTensor(1, 1, 3, 3).zero_()
                mask[0, 0, 0, 1] = 1.0
                mask[0, 0, 1, 0] = 1.0
                mask[0, 0, 2, 1] = 1.0
                mask[0, 0, 1, 2] = 1.0
                weight = self.conv_weight * Variable(mask)
            for j in range(self.max_steps - 1):
                tmp = F.conv2d(l_embs[-1], weight, padding=1)
                l_embs.append(tmp)

        landmarks = torch.cat(l_embs, 1)
        landmarks = landmarks.resize(batch_size, landmarks.size(1), 16).transpose(1, 2)

        logits = torch.bmm(landmarks, feature_msg.unsqueeze(-1)).squeeze(-1)
        prob = F.softmax(logits, dim=1)
        y_true = (ys[:, 0] * 4 + ys[:, 1]).squeeze()
        loss = self.loss(prob, y_true)
        acc = sum([1.0 for pred, target in zip(prob.max(1)[1].data.cpu().numpy(), y_true.data.cpu().numpy()) if
                   pred == target]) / batch_size
        return loss, acc

def eval_epoch(net, Xs, landmarks, ys, batch_sz, model_trainer, opt=None, use_cuda=False, dataname=None):
    loss, accs, total = 0.0, 0.0, 0.0

    for jj in range(0, len(Xs), batch_sz):
        X_batch, mask, landmark_batch, y_batch = create_batch(Xs[jj:jj + batch_sz], landmarks[jj:jj + batch_sz],
                                              ys[jj:jj + batch_sz], use_cuda=use_cuda)
        X_gen_batch, mask = model_trainer.create_localization_batch(dataname, jj, X_batch.size(0))
        l, acc = net.forward(X_gen_batch, mask, landmark_batch, y_batch)
        accs += acc
        total += 1
        loss += l.cpu().data.numpy()
        print('Batch: {}, Total: {}, loss: {}'.format(jj/batch_sz, int(len(Xs)/batch_sz), loss))
        if opt is not None:
            opt.zero_grad()
            l.backward()
            opt.step()
    return loss/total, accs/total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--full-dialogue', type='bool', default=False)
    parser.add_argument('--num-past-utterances', type=int, default=5)
    parser.add_argument('--condition-on-action', type='bool', default=False)
    parser.add_argument('--mask-conv', type='bool', default=False)
    parser.add_argument('--num-steps-location', type=int, default=2)
    parser.add_argument('--hidden-sz', type=int, default=256)
    parser.add_argument('--embed-sz', type=int, default=128)
    parser.add_argument('--batch-sz', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--exp-name', type=str, default='test')
    parser.add_argument('--log-time', type=float, default=2.,
                        help='how often to log training')
    parser.add_argument('--use-cuda', type='bool', default=True)
    parser.add_argument('--valid-patience', type=int, default=5)
    parser.add_argument('-mf', '--model-file', type=str, default='my_model')
    parser.add_argument('--localize-model-file', type=str, default='my_model')
    parser.add_argument('--resnet-features', type='bool', default=False)
    parser.add_argument('--fasttext-features', type='bool', default=False)
    parser.add_argument('--goldstandard-features', type='bool', default=True)
    parser.add_argument('--enc-emb-sz', type=int, default=32)
    parser.add_argument('--dec-emb-sz', type=int, default=32)
    parser.add_argument('--resnet-dim', type=int, default=2048)
    parser.add_argument('--resnet-proj-dim', type=int, default=64)
    parser.add_argument('--hsz', type=int, default=128)
    parser.add_argument('--bsz', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--bidirectional', type='bool', default=False)
    parser.add_argument('--attention', type=str, default='none')
    parser.add_argument('--pass-hidden-state', type='bool', default=True)
    parser.add_argument('--use-dec-state',type='bool', default=True)
    parser.add_argument('--rnn-type', type=str, default='LSTM')
    parser.add_argument('--use-prev-word', type='bool', default=True)
    parser.add_argument('--n-layers', type=int, default=1)
    parser.add_argument('--learningrate', type=float, default=.001)
    parser.add_argument('--min-word-freq', type=int, default=1)

    parser.add_argument('--dict-file', type=str, default='dict.txt')
    parser.add_argument('--temp-build', type='bool', default=False)
    parser.add_argument('--fill-padding-mask', type='bool', default=True)
    parser.add_argument('--min-sent-length', type=int, default=0)
    parser.add_argument('--load-data', type='bool', default=True)
    parser.add_argument('--num-steps', type=int, default=-1)
    parser.add_argument('--orientation-aware', type='bool', default=False,
                        help='if false, tourist is not orientation aware, \
                        act directions are not forward, turn left, etc; it\
                        is simply up, down, left, right')
    parser.add_argument('--sample-tokens', type='bool', default=False,
                        help='whether to sample next generated token')
    parser.add_argument('--split', type='bool', default=False,
                        help='whether to use split tokenizer when\
                        tokenizing messages (default is TweetTokenizer)')
    parser.add_argument('--beam-width', type=int, default=4,
                        help='width of beam search')
    parser.add_argument('--beam-search', type='bool', default=False,
                        help='Whether to use beam search when generating tokens')
    parser.set_defaults(data_dir='data/')


    args = parser.parse_args()
    model_trainer = TrainLanguageGenerator(args)
    model_trainer.load_model(args.model_file)
    args = model_trainer.args
    # for dataset_name in ['train', 'valid', 'test']:
    for dataset_name in ['train', 'valid', 'test']:
        model_trainer.load_localization_data(dataset_name,
                                             orientation_aware=False,
                                             full_dialogue=args.full_dialogue,
                                             num_past_utterances=args.num_past_utterances)

    exp_name = args.model_file + '_experiment' + str(time.time())
    exp_dir = os.path.join('experiment', exp_name)
    if os.path.exists(exp_dir):
        raise RuntimeError('Experiment directory {} already exist..'.format(exp_dir))
    os.mkdir(exp_dir)
    #
    logger = create_logger(os.path.join(exp_dir, 'log.txt'))
    logger.info(args)

    # data_dir = os.environ.get('TALKTHEWALK_DATADIR', './data')
    data_dir = './data'

    train_set = json.load(open(os.path.join(data_dir, 'talkthewalk.train.json')))
    valid_set = json.load(open(os.path.join(data_dir, 'talkthewalk.valid.json')))
    test_set = json.load(open(os.path.join(data_dir, 'talkthewalk.test.json')))

    dictionary = Dictionary('./data/{}'.format(args.dict_file), 1)

    neighborhoods = ['fidi', 'hellskitchen', 'williamsburg', 'uppereast', 'eastvillage']
    landmark_map = Landmarks(neighborhoods, include_empty_corners=True)

    train_Xs, train_landmarks, train_ys = load_data(train_set, landmark_map, dictionary, full_dialogue=args.full_dialogue, min_sent_length=args.min_sent_length, num_past_utterances=args.num_past_utterances)
    valid_Xs, valid_landmarks, valid_ys = load_data(valid_set, landmark_map, dictionary, full_dialogue=args.full_dialogue, min_sent_length=args.min_sent_length, num_past_utterances=args.num_past_utterances)
    test_Xs, test_landmarks, test_ys = load_data(test_set, landmark_map, dictionary, full_dialogue=args.full_dialogue, min_sent_length=args.min_sent_length, num_past_utterances=args.num_past_utterances)
    print('len of model_trainer x: {}'.format(len(model_trainer.train_localization_Xs)))
    print('len of train_xs x: {}'.format(len(train_Xs)))
    print('len of model_trainer valid x: {}'.format(len(model_trainer.valid_localization_Xs)))
    print('len of valid_xs x: {}'.format(len(valid_Xs)))
    print('len of model_trainer test x: {}'.format(len(model_trainer.test_localization_Xs)))
    print('len of test_xs x: {}'.format(len(test_Xs)))
    batch_sz = args.batch_sz
    hid_sz = 256
    emb_sz = 128
    use_cuda = True
    num_epochs = args.num_epochs

    net = LocationPredictor(emb_sz, hid_sz, len(dictionary), condition_on_action=args.condition_on_action,
                            mask_conv=args.mask_conv, max_steps=args.num_steps_location, use_cuda=use_cuda)

    if use_cuda:
        net = net.cuda()
    opt = optim.Adam(net.parameters())

    best_train_acc, best_val_acc, best_test_acc = 0.0, 0.0, 0.0
    for i in range(100):
        print('EPOCH NUMBER {}'.format(i))
        train_Xs, train_landmarks, train_ys = shuffle(train_Xs, train_landmarks, train_ys)
        train_loss, train_acc = eval_epoch(net, train_Xs, train_landmarks, train_ys, batch_sz, model_trainer, dataname='train', opt=opt, use_cuda=use_cuda)
        valid_loss, valid_acc = eval_epoch(net, valid_Xs, valid_landmarks, valid_ys, batch_sz, model_trainer, dataname='valid', use_cuda=use_cuda)
        test_loss, test_acc = eval_epoch(net, test_Xs, test_landmarks, test_ys, batch_sz, model_trainer, dataname='test', use_cuda=use_cuda)

        logger.info("Train loss: %.2f, Valid loss: %.2f, Test loss: %.2f" % (train_loss, valid_loss, test_loss))
        logger.info("Train acc: %.2f, Valid acc: %.2f, Test acc: %.2f" % (train_acc*100, valid_acc*100, test_acc*100))
        print("Train loss: %.2f, Valid loss: %.2f, Test loss: %.2f" % (train_loss, valid_loss, test_loss))
        print("Train acc: %.2f, Valid acc: %.2f, Test acc: %.2f" % (train_acc*100, valid_acc*100, test_acc*100))
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_train_acc, best_val_acc, best_test_acc = train_acc, valid_acc, test_acc


    logger.info(best_train_acc)
    logger.info(best_val_acc)
    logger.info(best_test_acc)
    print(best_train_acc)
    print(best_val_acc)
    print(best_test_acc)
    torch.save(net.state_dict(), args.localize_model_file)
    torch.save(opt.state_dict(), args.localize_model_file+'.optim')
    with open(args.localize_model_file+'.args', 'w') as f:
        json.dump(args, f)
