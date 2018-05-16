import argparse
import os
import ujson as json
import torch
import math
import torch.optim as optim
from collections import deque
import time
from sklearn.utils import shuffle
from data_loader import Landmarks, step_aware, load_features, \
    FasttextFeatures, GoldstandardFeatures, ResnetFeatures
from dict import Dictionary, START_TOKEN, END_TOKEN, UNK_TOKEN, PAD_TOKEN
from seq2seq import Seq2Seq
from kvmemnn import KVMemnn
from attrdict import AttrDict
import random
from utils import ProgressLogger


def str2bool(value):
    v = value.lower()
    if v in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_action(msg):
    msg_to_act = {'ACTION:TURNLEFT': 1,
                  'ACTION:TURNRIGHT': 2,
                  'ACTION:FORWARD': 3}
    return msg_to_act.get(msg, None)

def get_action_from_i(i):
    idx_to_act = ['LEFT', 'UP', 'RIGHT', 'DOWN']
    return idx_to_act[i-1]


# Determine if tourist went "up", "down", "left", "right"
def get_new_action(old_loc, new_loc):
    act_to_idx = {'LEFT': 1, 'UP': 2, 'RIGHT': 3, 'DOWN': 4, 'STAYED': -1}
    step_to_dir = {
        0: {
            1: 'N',
            -1: 'S',
            0: 'STAYED'
        },
        1: {
            0: 'E',
        },
        -1: {
            0: 'W'
        }
    }
    dir_to_act = {'N': 'UP', 'E': 'RIGHT', 'S': 'DOWN', 'W': 'LEFT', 'STAYED': 'STAYED'}

    step = [new_loc[0] - old_loc[0], new_loc[1] - old_loc[1]]
    direction = step_to_dir[step[0]][step[1]]
    return act_to_idx[dir_to_act[direction]]


class ActionObservationDictionary(object):
    """Just has the pad, end, and start indices for action/obs sequence"""
    def __init__(self, landmarks, actions, orientation_aware=True):
        if orientation_aware:
            self.pad_idx = len(landmarks) + len(actions)
            self.start_idx = self.pad_idx + 1
            self.end_idx = self.start_idx + 1
            self.unk_idx = self.end_idx + 1
            self.tok2i = {START_TOKEN: self.start_idx,
                          END_TOKEN: self.end_idx,
                          PAD_TOKEN: self.pad_idx,
                          UNK_TOKEN: self.unk_idx}
        else:
            self.tok2i = {
                PAD_TOKEN: 0,
                'LEFT': 1,
                'UP': 2,
                'RIGHT': 3,
                'DOWN': 4
                }


class TrainLanguageGenerator(object):
    """class for training the language generator. Provides a trainloop"""
    def setup_args(self):
        parser = argparse.ArgumentParser()
        parser.register('type', 'bool', str2bool)
        parser.add_argument('--log-time', type=float, default=2.,
                            help='how often to log training')
        parser.add_argument('--use-cuda', type='bool', default=True)
        parser.add_argument('--valid-patience', type=int, default=5)
        parser.add_argument('-mf', '--model-file', type=str, default='')
        parser.add_argument('--resnet-features', type='bool', default=False)
        parser.add_argument('--goldstandard-features', type='bool',
                            default=True)
        parser.add_argument('--num-steps', type=int, default=-1)
        parser.add_argument('--enc-emb-sz', type=int, default=32)
        parser.add_argument('--dec-emb-sz', type=int, default=32)
        parser.add_argument('--ctx-dim', type=int, default=0)
        parser.add_argument('--resnet-dim', type=int, default=2048)
        parser.add_argument('--resnet-proj-dim', type=int, default=64)
        parser.add_argument('--hsz', type=int, default=128)
        parser.add_argument('--num-epochs', type=int, default=500)
        parser.add_argument('--bsz', type=int, default=64)
        parser.add_argument('--exp-name', type=str, default='test')
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--bidirectional', type='bool', default=False)
        parser.add_argument('--attention', type=str, default='none')
        parser.add_argument('--pass-hidden-state', type='bool', default=True)
        parser.add_argument('--use-dec-state',type='bool', default=True)
        parser.add_argument('--rnn-type', type=str, default='LSTM')
        parser.add_argument('--use-prev-word', type='bool', default=True)
        parser.add_argument('--n-layers', type=int, default=1)
        parser.add_argument('--learningrate', type=float, default=1.e-4)
        parser.add_argument('--text_dict-file', type=str, default='text_dict.txt')
        parser.add_argument('--min-word-freq', type=int, default=3)
        parser.add_argument('--temp-build', type='bool', default=False)
        parser.add_argument('--fill-padding-mask', type='bool', default=True)
        parser.add_argument('--min-sent-length', type=int, default=0)
        parser.add_argument('--orientation-aware', type='bool', default=False,
                            help='if false, tourist is not orientation aware, \
                            act directions are not forward, turn left, etc; it\
                            is simply up, down, left, right')
        parser.add_argument('--sample-tokens', type='bool', default=False,
                            help='whether to sample next generated token')
        parser.add_argument('--split', type='bool', default=False,
                            help='whether to use split tokenizer when\
                            tokenizing messages (default is TweetTokenizer)')
        parser.add_argument('--beam-search', type='bool', default=False,
                            help='Whether to use beam search when \
                            generating tokens')
        parser.add_argument('--beam-width', type=int, default=4,
                            help='width of beam search')
        parser.add_argument('--use-actions', type='bool', default=True,
                            help='Whether to condition on actions')
        parser.set_defaults(data_dir='data/')
        self.args = parser.parse_args()

    def __init__(self, args=None):
        if args is None:
            self.setup_args()
            args = self.args
        else:
            self.args = args
        args_file = args.model_file.replace('.best_valid', '') + '.args'
        old_args = args
        if os.path.exists(args_file):
            print('Overriding args from {}'.format(args_file))
            args = self.override_args(args)
            self.args = args
        self.beam_width = old_args.beam_width
        self.beam_search = old_args.beam_search
        self.data_dir = args.data_dir

        self.num_epochs = args.num_epochs
        self.bsz = args.bsz
        self.contextlen = args.num_steps if args.num_steps >= 0 else None
        self.use_cuda = torch.cuda.is_available() and args.use_cuda
        self.valid_patience = args.valid_patience
        self.model_file = args.model_file
        self.log_time = args.log_time
        self.learning_rate = args.learningrate
        self.min_sent_length = args.min_sent_length
        self.orientation_aware = args.orientation_aware
        self.min_word_freq = args.min_word_freq
        self.dict_file = args.dict_file
        self.condition_on_action = args.use_actions
        self.logger = ProgressLogger(should_humanize=False, throttle=0.1)

        self.neighborhoods = ['fidi', 'hellskitchen', 'williamsburg',
                              'uppereast', 'eastvillage']
        self.landmark_map = Landmarks(self.neighborhoods,
                                      include_empty_corners=False)
        self.dictionary = Dictionary(self.data_dir+self.dict_file,
                                     self.min_word_freq,
                                     split=args.split)
        self.action_obs_dict = ActionObservationDictionary(
                                    self.landmark_map.i2landmark,
                                    [1, 2, 3],
                                    orientation_aware=self.orientation_aware)
        print('Loading Datasets...')
        self.load_datasets()
        self.setup_feature_loaders()
        print('Building Train Data...')
        self.train_data = self.load_data(self.train_set,
                                         'train',
                                         temp_build=args.temp_build,
                                         orientation_aware=self.orientation_aware)
        print('Building Valid Data...')
        self.valid_data = self.load_data(self.valid_set,
                                         'valid',
                                         temp_build=args.temp_build,
                                         orientation_aware=self.orientation_aware)
        print('Building Test Data...')
        self.test_data = self.load_data(self.test_set,
                                        'test',
                                        temp_build=args.temp_build,
                                        orientation_aware=self.orientation_aware)
        self.setup_model()

    def override_args(self, args):
        args = AttrDict(vars(args))
        args_file = args.model_file.replace('.best_valid', '') + '.args'
        with open(args_file) as f:
            new_args = json.load(f)
            for key, val in new_args.items():
                if key == 'cuda':
                    args['use_cuda'] = val
                    args['cuda'] = val
                else:
                    args[key] = val
        return args

    def setup_model(self):
        self.max_len = max([len(seq) for seq in self.train_data[0]])
        self.model = Seq2Seq(self.args,
                             n_lands=10,
                             n_acts=3 if self.orientation_aware else 4,
                             n_words_trg=len(self.dictionary),
                             vocab_src=self.action_obs_dict,
                             vocab_trg=self.dictionary,
                             max_length=self.max_len)

        # self.model = KVMemnn(args, self.dictionary)
        if self.use_cuda:
            self.model.cuda()
        self.optim = optim.Adam(filter(lambda p: p.requires_grad,
                                       self.model.parameters()),
                                lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                self.optim,
                                factor=0.5,
                                patience=2,
                                verbose=True)

    def load_datasets(self):
        dataset_names = ['train', 'valid', 'test']
        datasets = []
        for dataset in dataset_names:
            dataset_path = os.path.join(self.data_dir,
                                        'talkthewalk.{}.json'.format(dataset))
            with open(dataset_path) as f:
                datasets.append(json.load(f))
        self.train_set = datasets[0]
        self.valid_set = datasets[1]
        self.test_set = datasets[2]

    def setup_feature_loaders(self):
        self.feature_loaders = {}
        if self.args.resnet_features:
            self.feature_loaders['resnet'] = ResnetFeatures(
                                            os.path.join(self.data_dir,
                                                         'resnetfeat.json'))
        if self.args.goldstandard_features:
            self.feature_loaders['goldstandard'] = GoldstandardFeatures(
                                                            self.landmark_map,
                                                            orientation_aware=self.orientation_aware)

    def load_data(self, dataset, dataset_name, temp_build=False,
                  orientation_aware=True):
        Xs = []         # x_i = [a_1, o_1, a_2, ..., a_n, o_n] acts + obs
        ys = []         # y_i = msg from tourist
        feature_loaders = self.feature_loaders
        dataset_path = os.path.join(self.data_dir,
                                    "{}_NLG_data_mwf-{}_msl-{}_dict-{}_oa-{}_contextlen-{}/".format(
                                     dataset_name,
                                     self.min_word_freq,
                                     self.min_sent_length,
                                     self.dict_file,
                                     self.orientation_aware,
                                     self.contextlen))
        if os.path.exists(dataset_path) and not temp_build:
            data = []
            for d in ['Xs', 'ys']:
                print("Loading {} for {}".format(d, dataset_name))
                f_name = '{}.json'.format(d)
                with open(os.path.join(dataset_path, f_name)) as f:
                    data.append(json.load(f))
            return data
        else:
            for j in range(len(dataset)):
                config = dataset[j]
                self.logger.log(j, len(dataset))
                loc = config['start_location']
                boundaries = config['boundaries']
                neighborhood = config['neighborhood']
                act_obs_memory = deque(maxlen=self.contextlen)
                for msg in config['dialog']:
                    if msg['id'] == 'Tourist':
                        act = get_action(msg['text'])
                        if act is None:
                            msg_length = len(msg['text'].split(' '))
                            if msg_length > self.min_sent_length:
                                y = self.dictionary.encode(msg['text'],
                                                           include_end=True)
                                if len(act_obs_memory) == 0:
                                    obs_emb = {}
                                    for k, loader in feature_loaders.items():
                                        if k == 'goldstandard':
                                            features = loader.get(neighborhood,
                                                                  loc)
                                        else:
                                            features = loader.get(neighborhood,
                                                                  loc[0],
                                                                  loc[1])
                                        obs_emb[k] = features
                                    act_obs_memory.append(obs_emb)
                                if orientation_aware and self.condition_on_action:
                                    act_obs_memory.append(
                                        self.action_obs_dict.tok2i[END_TOKEN])
                                Xs.append(list(act_obs_memory))
                                ys.append(y)
                                act_obs_memory.clear()
                            # else:
                            #     act_obs_memory.clear()
                        else:
                            new_loc = step_aware(act-1, loc, boundaries)
                            old_loc = loc
                            loc = new_loc
                            if orientation_aware and self.condition_on_action:
                                act_obs_memory.append(act)
                            if act == 3:  # went forward
                                if not orientation_aware:
                                    act_dir = get_new_action(old_loc, new_loc)
                                    if act_dir != -1 and self.condition_on_action:
                                        act_obs_memory.append(act_dir)
                                if orientation_aware or act_dir != -1:
                                    obs_emb = {}
                                    for k, loader in feature_loaders.items():
                                        if k == 'goldstandard':
                                            features = loader.get(neighborhood,
                                                                  loc)
                                        else:
                                            features = loader.get(neighborhood,
                                                                  loc[0],
                                                                  loc[1])
                                        obs_emb[k] = features
                                    act_obs_memory.append(obs_emb)

            data = [Xs, ys]
            if not temp_build:
                print("Finished building {}, saving now".format(dataset_name))
                os.makedirs(dataset_path)
                for i, d in enumerate(['Xs', 'ys']):
                    print("Saving {}".format(d))
                    with open(os.path.join(dataset_path, '{}.json'.format(d)), 'w') as f:
                        json.dump(data[i], f)
        return data

    def load_localization_data(self, dataset_name,
                               orientation_aware=True, full_dialogue=False,
                               num_past_utterances=5, save_data=False,
                               temp_build=True):
        if dataset_name == 'train':
            dataset = self.train_set
        elif dataset_name == 'valid':
            dataset = self.valid_set
        else:
            dataset = self.test_set
        dataset_path = os.path.join(self.data_dir,
                                    "{}_NLG_localize_data_mwf-{}_msl-{}_dict-{}_oa-{}_num_past_utt-{}_full_dialoge-{}_beam_search-{}_bs_width-{}/".format(
                                     dataset_name,
                                     self.min_word_freq,
                                     self.min_sent_length,
                                     self.dict_file,
                                     orientation_aware,
                                     num_past_utterances,
                                     full_dialogue,
                                     self.beam_search,
                                     self.beam_width
                                     ))
        if os.path.exists(dataset_path) and not temp_build:
            print('Localization data already built; loading {} split'.format(dataset_name))
            f_name = 'Xs.json'
            with open(os.path.join(dataset_path, f_name)) as f:
                Xs = json.load(f)
            if dataset_name == 'train':
                self.train_localization_Xs = Xs
            elif dataset_name == 'valid':
                self.valid_localization_Xs = Xs
            else:
                self.test_localization_Xs = Xs
            return Xs
        print('Building localization data for {} split'.format(dataset_name))
        Xs = []         # x_i = [a_1, o_1, a_2, ..., a_n, o_n] acts + obs
        dialogue = []
        for j in range(len(dataset)):
            self.logger.log(j, len(dataset))
            config = dataset[j]
            loc = config['start_location']
            boundaries = config['boundaries']
            neighborhood = config['neighborhood']
            act_obs_memory = deque(maxlen=self.contextlen)
            for msg in config['dialog']:
                if msg['id'] == 'Tourist':
                    act = get_action(msg['text'])
                    if act is None:
                        msg_length = len(msg['text'].split(' '))
                        if msg_length > self.min_sent_length:
                            y = self.dictionary.encode(msg['text'],
                                                       include_end=True)
                            obs_emb = {}
                            for k, loader in self.feature_loaders.items():
                                if k == 'goldstandard':
                                    features = loader.get(neighborhood,
                                                          loc)
                                else:
                                    features = loader.get(neighborhood,
                                                          loc[0],
                                                          loc[1])
                                obs_emb[k] = features
                            act_obs_memory.append(obs_emb)
                            if orientation_aware:
                                act_obs_memory.append(
                                    self.action_obs_dict.tok2i[END_TOKEN])
                            X = list(act_obs_memory)
                            data = self.create_batch([X], [y])
                            X_batch, _, _, X_lengths, _, max_len = data
                            # X_batch, _, X_lengths, _, max_len = data
                            res = self.model.forward(src_var=X_batch,
                                                     src_lengths=X_lengths,
                                                     trg_var=None,
                                                     trg_lengths=None,
                                                     max_length=max_len,
                                                     return_attention=True)
                            pred = res['preds'][0, :].tolist()
                            import pdb; pdb.set_trace()
                            dialogue.append(self.dictionary.encode(self.dictionary.decode(pred), include_end=False))
                            utt = [y for x in dialogue[-num_past_utterances:] for y in x] + [self.dictionary[END_TOKEN]]
                            Xs.append(utt)
                            act_obs_memory.clear()
                        else:
                            act_obs_memory.clear()
                    else:
                        new_loc = step_aware(act-1, loc, boundaries)
                        old_loc = loc
                        loc = new_loc
                        if orientation_aware:
                            act_obs_memory.append(act)
                        if act == 3:  # went forward
                            if not orientation_aware:
                                act_dir = get_new_action(old_loc, new_loc)
                                if act_dir != -1:
                                    act_obs_memory.append(act_dir)
                            if orientation_aware or act_dir != -1:
                                obs_emb = {}
                                for k, loader in self.feature_loaders.items():
                                    if k == 'goldstandard':
                                        features = loader.get(neighborhood, loc)
                                    else:
                                        features = loader.get(neighborhood, loc[0], loc[1])
                                    obs_emb[k] = features
                                act_obs_memory.append(obs_emb)
                elif full_dialogue:
                    dialogue.append(self.dictionary.encode(msg['text'],
                                              include_end=False))
        if save_data:
            f_name = 'Xs.json'
            os.makedirs(dataset_path)
            with open(os.path.join(dataset_path, f_name), 'w') as f:
                json.dump(Xs, f)
        if dataset_name == 'train':
            self.train_localization_Xs = Xs
        elif dataset_name == 'valid':
            self.valid_localization_Xs = Xs
        else:
            self.test_localization_Xs = Xs

        return Xs

    def create_batch(self, Xs, ys):
        batch_size = len(Xs)
        seq_lens = [len(seq) for seq in Xs]
        y_lens = [len(y) for y in ys]
        max_y_len = max(y_lens)
        max_X_len = max(seq_lens)
        X_batch = [[self.action_obs_dict.tok2i[PAD_TOKEN] for _ in range(max_X_len)] for _ in range(batch_size)]
        mask = torch.FloatTensor(batch_size, max_X_len).zero_()
        for i, seq in enumerate(Xs):
            for j, elem in enumerate(seq):
                X_batch[i][j] = elem
            mask[i, :len(seq)] = 1.0
        y_batch = torch.LongTensor(batch_size, max_y_len).fill_(
                                                    self.dictionary[PAD_TOKEN])
        for i, seq in enumerate(ys):
            y_batch[i, :len(seq)] = torch.LongTensor(seq)

        # Sort batch according to length of sequence
        sorted_seq_lens, sorted_indices = torch.sort(
            torch.LongTensor(seq_lens),
            descending=True)
        sorted_X_batch = [[self.action_obs_dict.tok2i[PAD_TOKEN] for _ in range(max_X_len)] for _ in range(batch_size)]
        sorted_y_batch = torch.LongTensor(batch_size, max_y_len).fill_(
            self.dictionary[PAD_TOKEN]
        )
        sorted_mask = torch.FloatTensor(batch_size, max_X_len).zero_()
        sorted_y_lens = []
        i = 0
        for idx in sorted_indices:
            sorted_X_batch[i][:] = X_batch[idx][:]
            sorted_y_batch[i, :] = y_batch[idx][:]
            sorted_y_lens.append(y_lens[i])
            sorted_mask[i, :sorted_seq_lens[i]] = 1.0
            i += 1

        if self.use_cuda:
            sorted_mask = sorted_mask.cuda()
            sorted_y_batch = sorted_y_batch.cuda()
        return (sorted_X_batch,
                sorted_mask,
                sorted_y_batch,
                sorted_seq_lens,
                sorted_y_lens,
                max_y_len)

    def create_localization_batch(self, dataname, start_idx, batch_sz):
        if dataname == 'train':
            Xs = self.train_localization_Xs
        elif dataname == 'valid':
            Xs = self.valid_localization_Xs
        else:
            Xs = self.test_localization_Xs

        seq_lens = [len(seq) for seq in Xs[start_idx: start_idx+batch_sz]]
        max_X_len = max(seq_lens)
        X_batch = torch.LongTensor(batch_sz, max_X_len).fill_(self.dictionary[PAD_TOKEN])
        mask = torch.Tensor(X_batch.size()).fill_(0)

        # X_batch = [[0 for _ in range(max_X_len)] for _ in range(batch_size)]
        for i, seq in enumerate(Xs[start_idx: start_idx+batch_sz]):
            for j, elem in enumerate(seq):
                X_batch[i, j] = elem
                if elem != self.dictionary[PAD_TOKEN]:
                    mask[i, j] = 1.0
        if self.use_cuda:
            X_batch = X_batch.cuda()
            mask = mask.cuda()
        return X_batch, mask

    def train(self, num_epochs=None):
        print("Beginning Training...")
        if num_epochs is None:
            num_epochs = self.num_epochs
        Xs, ys = self.train_data
        best_valid = float('inf')
        best_test = float('inf')
        best_train_loss_at_best_valid = float('inf')
        valid_patience = 0

        to_log = time.time()
        start = time.time()
        for epoch_num in range(self.num_epochs):
            Xs, ys = shuffle(Xs, ys)
            total_loss, total_loss_since_log, total, total_since_log = 0.0, 0.0, 0.0, 0.0
            batch_num = 0
            for jj in range(0, len(Xs), self.bsz):
                batch_num += 1
                data = self.create_batch(Xs[jj:jj + self.bsz],
                                         ys[jj:jj + self.bsz])
                X_batch, mask, y_batch, X_lengths, y_lengths, max_len = data

                res = self.model.forward(src_var=X_batch,
                                         src_lengths=X_lengths,
                                         trg_var=y_batch,
                                         trg_lengths=y_lengths,
                                         max_length=max_len,
                                         encoder_mask=mask,
                                         return_attention=True,
                                         train=True)
                total += 1
                total_since_log += 1
                loss = res['loss']['loss']
                total_loss += loss.cpu().data.numpy()
                total_loss_since_log += loss.cpu().data.numpy()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if time.time() - to_log >= self.log_time:
                    elapsed = time.time() - start
                    print('Elapsed_time: {}, Batch: {}/{}; loss: {:.2f}; '.format(int(elapsed), batch_num, int(len(Xs)/self.bsz), total_loss_since_log/total_since_log))
                    to_log = time.time()
                    pred = res['preds'][0, :]
                    found_end = False

                    probs = res['log_probs'][0, :, :]
                    log_prob_pred = torch.gather(probs, 1, pred.unsqueeze(1))
                    for i in range(pred.size(0)):
                        if found_end:
                            log_prob_pred[i] = 0
                        elif pred[i] == self.dictionary[END_TOKEN]:
                            found_end = True
                    prob = torch.exp(log_prob_pred).prod()
                    print('teacher forced: {}'.format(res['teacher_force']))
                    print('target: {}'.format(self.dictionary.decode(y_batch[0, :])))
                    print('generate: {}'.format(self.dictionary.decode(pred)))
                    print('probability: {}'.format(prob))
                    # print('target: {}'.format(y_batch[0, :]))
                    # print('generate: {}'.format(pred))
                    print('\n')
                    total_loss_since_log = 0
                    total_since_log = 0
            print('Epoch: {}, Loss: {}'.format(epoch_num, total_loss/total))
            valid_loss = self.eval_epoch()
            self.lr_scheduler.step(valid_loss)
            if valid_loss < best_valid:
                print('New Best Valid: {}'.format(valid_loss))
                best_test = self.eval_test()
                print('Test Loss at Best Valid: {}'.format(best_test))
                best_train_loss_at_best_valid = total_loss/total
                print('Train Loss at Best Valid: {}'.format(best_train_loss_at_best_valid))
                self.save_valid()
                best_valid = valid_loss
                valid_patience = 0
            else:
                valid_patience += 1
                print("BEST VALID STILL GOOD AFTER {} EPOCHS".format(valid_patience))
                if valid_patience == self.valid_patience:
                    print("Finished training; saving model to {}".format(self.model_file))
                    self.save_model()
                    test_loss = self.eval_test()
                    print('Test Loss at final epoch: {}'.format(test_loss))

                    print('Best Valid Loss: {}'.format(best_valid))
                    print('Best Train Loss at Valid: {}'.format(best_train_loss_at_best_valid))
                    print('Best Test Loss at Valid: {}'.format(best_test))
                    return

        print('Finished {} epochs; saving anyway...'.format(self.num_epochs))
        self.save_model()
        val_loss = self.eval_epoch()
        print('Validation Loss last epoch: {}'.format(val_loss))
        test_loss = self.eval_test()
        print('Test Loss last epoch: {}'.format(test_loss))
        print('Best Valid Loss: {}'.format(best_valid))
        print('Best Train Loss at Valid: {}'.format(best_train_loss_at_best_valid))
        print('Best Test Loss at Valid: {}'.format(best_test))


    def eval_epoch(self):
        Xs, ys = self.valid_data
        Xs, ys = shuffle(Xs, ys)
        total_loss, total = 0.0, 0.0
        batch_num = 0
        for jj in range(0, len(Xs), self.bsz):
            batch_num += 1
            data = self.create_batch(Xs[jj:jj + self.bsz],
                                     ys[jj:jj + self.bsz])
            X_batch, mask, y_batch, X_lengths, y_lengths, max_len = data
            res = self.model.forward(src_var=X_batch,
                                     src_lengths=X_lengths,
                                     trg_var=y_batch,
                                     trg_lengths=None,
                                     max_length=max_len,
                                     return_attention=True)
            total += 1
            loss = res['loss']
            total_loss += loss['loss'].cpu().data.numpy()
        return total_loss/total

    def eval_test(self):
        Xs, ys = self.test_data
        Xs, ys = shuffle(Xs, ys)
        total_loss, total = 0.0, 0.0
        batch_num = 0
        for jj in range(0, len(Xs), self.bsz):
            batch_num += 1
            data = self.create_batch(Xs[jj:jj + self.bsz],
                                     ys[jj:jj + self.bsz])
            X_batch, mask, y_batch, X_lengths, y_lengths, max_len = data
            res = self.model.forward(src_var=X_batch,
                                     src_lengths=X_lengths,
                                     trg_var=y_batch,
                                     trg_lengths=None,
                                     max_length=max_len,
                                     return_attention=True)
            total += 1
            loss = res['loss']
            total_loss += loss['loss'].cpu().data.numpy()
        return total_loss/total

    def predict(self, data_name, num_preds, start=None, print_preds=True, one_batch=False, use_cuda=False, max_len=None):
        if data_name == 'train':
            data = self.train_data
        elif data_name == 'test':
            data = self.test_data
        else:
            data = self.valid_data

        Xs, ys = data

        if start is None:
            start = random.choice(range(len(Xs)-num_preds))

        for jj in range(start, len(Xs), num_preds):
            data = self.create_batch(Xs[jj:jj + num_preds],
                                     ys[jj:jj + num_preds])
            X_batch, mask, y_batch, X_lengths, y_lengths, max_d_len = data
            if max_len is None:
                max_len = max_d_len
            res = self.model.forward(src_var=X_batch,
                                     src_lengths=X_lengths,
                                     trg_var=None,
                                     trg_lengths=None,
                                     max_length=max_len,
                                     return_attention=True)
            preds = res['preds']
            probs = res['log_probs']
            if print_preds:
                for i in range(num_preds):
                    pred = preds[i, :]
                    # prob = math.exp(probs[i])
                    obs = X_batch[i]
                    print_obs = []
                    for a_or_o in obs:
                        if type(a_or_o) is int:
                            if a_or_o == 0:
                                break
                            print_obs.append(get_action_from_i(a_or_o))
                        else:
                            print_obs.append([self.landmark_map.i2landmark[k-1] for kk in a_or_o.values() for k in kk])
                    print('observation: {}'.format(print_obs))
                    print('target: {}'.format(self.dictionary.decode(y_batch[i, :])))
                    print('generate: {}'.format(self.dictionary.decode(pred)))
                    # print('probability: {}'.format(prob))
                    print('\n')
                break
            if one_batch:
                if not use_cuda:
                    return preds.cpu()
                else:
                    return preds
        return preds

    def predict_single(self, data_name, idx):
        if data_name == 'train':
            data = self.train_data
        elif data_name == 'test':
            data = self.test_data
        else:
            data = self.valid_data

        Xs, ys = data
        data = self.create_batch(Xs[idx:idx + 1],
                                 ys[idx:idx + 1])
        X_batch, mask, y_batch, X_lengths, y_lengths, max_len = data
        res = self.model.forward(src_var=X_batch,
                                 src_lengths=X_lengths,
                                 trg_var=None,
                                 trg_lengths=None,
                                 max_length=max_len,
                                 return_attention=True)
        preds = res['preds']

        return preds[0, :]

    def load_model(self, model_file):
        if os.path.exists(model_file):
            print('IT EXISTS')
            self.model.load_state_dict(torch.load(model_file))
            if os.path.exists(model_file + '.optim'):
                self.optim.load_state_dict(torch.load(model_file + '.optim'))
        else:
            print("IT DOES NOT EXIST")

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_file)
        torch.save(self.optim.state_dict(), self.model_file+'.optim')
        with open(self.model_file+'.args', 'w') as f:
            json.dump(self.args, f)

    def save_valid(self):
        torch.save(self.model.state_dict(), self.model_file + '.best_valid')
        torch.save(self.optim.state_dict(), self.model_file+'.optim_best_valid')


if __name__ == '__main__':
    trainer = TrainLanguageGenerator()
    trainer.load_model(trainer.model_file)

    # trainer.train()
    print("TRAIN DATA")
    trainer.predict('train', 20)
    print("VALID DATA")
    trainer.predict('valid', 20)
    print("TEST DATA")
    trainer.predict('test', 20)
    # trainer.train()
    # trainer.test_predict()
    # import pdb; pdb.set_trace()

    # trainer.eval_epoch()
    # trainer.eval_test()
