import argparse
import os
import ujson as json
import torch
import torch.optim as optim
from collections import deque
import time
from sklearn.utils import shuffle
from torch.autograd import Variable
from data_loader import Landmarks, step_aware, load_features, \
    FasttextFeatures, GoldstandardFeatures, ResnetFeatures
from dict import Dictionary, START_TOKEN, END_TOKEN, UNK_TOKEN, PAD_TOKEN
from seq2seq import Seq2Seq
from kvmemnn import KVMemnn
from attrdict import AttrDict
import random

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


def to_variable(obj, cuda=False):
    if torch.is_tensor(obj):
        var = Variable(obj)
        if cuda:
            var = var.cuda()
        return var
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [to_variable(x, cuda=cuda) for x in obj]
    if isinstance(obj, dict):
        return {k: to_variable(v, cuda=cuda) for k, v in obj.items()}


class ActionObservationDictionary(object):
    """Just has the pad, end, and start indices for action/obs sequence"""
    def __init__(self, landmarks, actions):
        self.pad_idx = len(landmarks) + len(actions)
        self.start_idx = self.pad_idx + 1
        self.end_idx = self.start_idx + 1
        self.unk_idx = self.end_idx + 1
        self.tok2i = {START_TOKEN: self.start_idx,
                      END_TOKEN: self.end_idx,
                      PAD_TOKEN: self.pad_idx,
                      UNK_TOKEN: self.unk_idx}


class TrainLanguageGenerator(object):
    """class for training the language generator. Provides a trainloop"""
    def setup_args(self):
        parser = argparse.ArgumentParser()
        parser.register('type', 'bool', str2bool)
        parser.add_argument('--log-time', type=float, default=2.,
                            help='how often to log training')
        parser.add_argument('--use-cuda', type='bool', default=True)
        parser.add_argument('--valid-patience', type=int, default=5)
        parser.add_argument('-mf', '--model-file', type=str, default='my_model')
        parser.add_argument('--resnet-features', type='bool', default=True)
        parser.add_argument('--fasttext-features', type='bool', default=False)
        parser.add_argument('--goldstandard-features', type='bool', default=True)
        parser.add_argument('--num-steps', type=int, default=-1)
        parser.add_argument('--enc-emb-sz', type=int, default=32)
        parser.add_argument('--dec-emb-sz', type=int, default=32)
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
        # parser.add_argument('--n-enc-layers', type=int, default=2)
        # parser.add_argument('--n-dec-layers', type=int, default=2)
        parser.add_argument('--n-layers', type=int, default=1)
        parser.add_argument('--learningrate', type=float, default=.001)
        parser.add_argument('--dict-file', type=str, default='dict.txt')
        parser.add_argument('--temp-build', type='bool', default=False)
        parser.add_argument('--fill-padding-mask', type='bool', default=False)
        parser.add_argument('--min-sent-length', type=int, default=0)
        parser.add_argument('--load-data', type='bool', default=True)


        parser.set_defaults(data_dir='data/',
                            goldstandard_features=True,
                            resnet_features=True,
                            fill_padding_mask=True,
                            # bidirectional=False,
                            # pass_hidden_state=True,
                            # use_dec_state=True,
                            # use_prev_word=True,
                            # cuda=False,
                            )
        self.args = parser.parse_args()


    def __init__(self, args=None):
        if args is None:
            self.setup_args()
            args = self.args
        else:
            self.args = args
        if os.path.exists(args.model_file + '.args'):
            args = self.override_args(args)
            self.args = args
        self.data_dir = args.data_dir
        self.enc_emb_sz = args.enc_emb_sz
        self.dec_emb_sz = args.dec_emb_sz
        self.resnet_dim = args.resnet_dim
        self.resnet_proj_dim = args.resnet_proj_dim
        self.hsz = args.hsz
        self.num_epochs = args.num_epochs
        self.bsz = args.bsz
        self.contextlen = args.num_steps if args.num_steps >= 0 else None
        self.bidirectional = args.bidirectional
        self.attention = args.attention
        self.pass_hidden_state = args.pass_hidden_state
        self.rnn_type = args.rnn_type
        self.use_prev_word = args.use_prev_word
        self.use_dec_state = args.use_dec_state
        # self.n_enc_layers = args.n_enc_layers
        # self.n_dec_layers = args.n_dec_layers
        self.n_layers = args.n_layers
        self.dropout = args.dropout
        self.use_cuda = torch.cuda.is_available() and args.use_cuda
        self.valid_patience = args.valid_patience
        self.model_file = args.model_file
        self.log_time = args.log_time
        self.learning_rate = args.learningrate
        self.min_sent_length = args.min_sent_length

        self.neighborhoods = ['fidi', 'hellskitchen', 'williamsburg',
                              'uppereast', 'eastvillage']
        self.landmark_map = Landmarks(self.neighborhoods,
                                      include_empty_corners=True)
        self.dictionary = Dictionary(self.data_dir+args.dict_file, 1)

        self.action_obs_dict = ActionObservationDictionary(self.landmark_map.itos, [1, 2, 3])
        print('Loading Datasets...')
        self.load_datasets()
        self.setup_feature_loaders()


        print('Building Train Data...')
        self.train_data = self.load_data(self.train_set,
                                         'train_gold+resnet',
                                         self.feature_loaders,
                                         temp_build=args.temp_build)
        if args.load_data:
            print('Building Valid Data...')
            self.valid_data = self.load_data(self.valid_set,
                                             'valid_gold+resnet',
                                             self.feature_loaders,
                                             temp_build=args.temp_build)
            print('Building Test Data...')
            self.test_data = self.load_data(self.test_set,
                                            'test_gold+resnet',
                                            self.feature_loaders,
                                            temp_build=args.temp_build)
        self.setup_model()

    def override_args(self, args):
        args = AttrDict(vars(args))
        with open(args.model_file + '.args') as f:
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
        self.model = Seq2Seq(n_lands=11,
                             n_acts=3,
                             n_words_trg=len(self.dictionary),
                             hidden_size=self.hsz,
                             enc_emb_dim=self.enc_emb_sz,
                             dec_emb_dim=self.dec_emb_sz,
                             resnet_dim=self.resnet_dim,
                             resnet_proj_dim=self.resnet_proj_dim,
                             n_enc_layers=self.n_layers,
                             n_dec_layers=self.n_layers,
                             dropout=self.dropout,
                             word_dropout=self.dropout,
                             bidirectional=self.bidirectional,
                             attn_type=self.attention,
                             pass_hidden_state=self.pass_hidden_state,
                             vocab_src=self.action_obs_dict,
                             vocab_trg=self.dictionary,
                             rnn_type=self.rnn_type,
                             ctx_dim=0,
                             use_prev_word=self.use_prev_word,
                             use_dec_state=True,
                             max_length=self.max_len,
                             cuda=self.use_cuda,
                             use_resnet=self.args.resnet_features,
                             fill_padding_mask=self.args.fill_padding_mask)

        # self.model = KVMemnn(args, self.dictionary)
        if self.use_cuda:
            self.model.cuda()
        self.optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, factor=0.5, patience=2, verbose=True)

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
        if self.args.fasttext_features:
            textfeatures = load_features(self.neighborhoods)
            self.feature_loaders['fasttext'] = FasttextFeatures(
                                        textfeatures,
                                        '/private/home/harm/data/wiki.en.bin')
        if self.args.resnet_features:
            self.feature_loaders['resnet'] = ResnetFeatures(
                                            os.path.join(self.data_dir,
                                                         'resnetfeat.json'))
        if self.args.goldstandard_features:
            self.feature_loaders['goldstandard'] = GoldstandardFeatures(
                                                            self.landmark_map)

    def load_data(self, dataset, dataset_name, feature_loaders, temp_build=False):
        Xs = []         # x_i = [a_1, o_1, a_2, ..., a_n, o_n] acts + obs
        tourist_locs = []
        landmarks = []
        ys = []         # y_i = msg from tourist
        dataset_path = os.path.join(self.data_dir, "{}_NLG_data/".format(dataset_name))
        if os.path.exists(dataset_path) and not temp_build:
            data = []
            for d in ['Xs', 'tourist_locs', 'landmarks', 'ys']:
                print("Loading {} for {}".format(d, dataset_name))
                with open(os.path.join(dataset_path, '{}.json'.format(d))) as f:
                    data.append(json.load(f))
            return data
        else:
            for config in dataset:
                loc = config['start_location']
                boundaries = config['boundaries']
                neighborhood = config['neighborhood']
                act_obs_memory = deque(maxlen=self.contextlen)
                for msg in config['dialog']:
                    if msg['id'] == 'Tourist':
                        act = get_action(msg['text'])
                        if act is None:
                            if len(msg['text'].split(' ')) > self.min_sent_length:
                                y = self.dictionary.encode(msg['text'], include_end=True)
                                ls, tourist_loc = self.landmark_map.get_landmarks_2d(
                                                neighborhood, boundaries, loc)
                                landmarks.append(ls)
                                obs_emb = {}
                                for k, loader in feature_loaders.items():
                                    if k == 'goldstandard':
                                        features = loader.get(neighborhood, loc)
                                    else:
                                        features = loader.get(neighborhood, loc[0], loc[1])
                                    obs_emb[k] = features
                                # obs_emb = feature_loader.get(neighborhood, loc)
                                act_obs_memory.append(obs_emb)

                                Xs.append(list(act_obs_memory) + [self.action_obs_dict.tok2i[END_TOKEN]])
                                ys.append(y)
                                tourist_locs.append(tourist_loc)
                                act_obs_memory.clear()
                            else:
                                act_obs_memory.clear()
                        else:
                            loc = step_aware(act, loc, boundaries)
                            act_obs_memory.append(act)
                            if act == 2:  # went forward
                                ls, _ = self.landmark_map.get_landmarks_2d(
                                                neighborhood, boundaries, loc)
                                landmarks.append(ls)
                                obs_emb = {}
                                for k, loader in feature_loaders.items():
                                    if k == 'goldstandard':
                                        features = loader.get(neighborhood, loc)
                                    else:
                                        features = loader.get(neighborhood, loc[0], loc[1])
                                    obs_emb[k] = features
                                # obs_emb = feature_loader.get(neighborhood, loc)
                                act_obs_memory.append(obs_emb)

            data = [Xs, tourist_locs, landmarks, ys]
            if not temp_build:
                print("Finished building {}, saving now".format(dataset_name))
                os.makedirs(dataset_path)
                for i, d in enumerate(['Xs', 'tourist_locs', 'landmarks', 'ys']):
                    print("Saving {}".format(d))
                    with open(os.path.join(dataset_path, '{}.json'.format(d)), 'w') as f:
                        json.dump(data[i], f)
        return data

    def create_batch(self, Xs, tourist_locs, ys):
        batch_size = len(Xs)
        seq_lens = [len(seq) for seq in Xs]
        y_lens = [len(y) for y in ys]
        max_y_len = max(y_lens)
        max_X_len = max(seq_lens)
        X_batch = [[0 for _ in range(max_X_len)] for _ in range(batch_size)]
        mask = torch.FloatTensor(batch_size, max_X_len).zero_()
        for i, seq in enumerate(Xs):
            for j, elem in enumerate(seq):
                X_batch[i][j] = elem
            mask[i, :len(seq)] = 1.0
        y_batch = torch.LongTensor(batch_size, max_y_len).fill_(self.dictionary[PAD_TOKEN])
        for i, seq in enumerate(ys):
            y_batch[i, :len(seq)] = torch.LongTensor(seq)
        # y_batch = torch.LongTensor(ys)
        tourist_loc_batch = torch.LongTensor(tourist_locs)

        # Sort batch according to length of sequence
        sorted_seq_lens, sorted_indices = torch.sort(
            torch.LongTensor(seq_lens),
            descending=True)
        sorted_X_batch = [[self.action_obs_dict.tok2i[PAD_TOKEN] for _ in range(max_X_len)] for _ in range(batch_size)]
        sorted_y_batch = torch.LongTensor(batch_size, max_y_len).zero_()
        sorted_tourist_loc_batch = torch.LongTensor(tourist_loc_batch.size())
        sorted_mask = torch.FloatTensor(batch_size, max_X_len).zero_()
        sorted_y_lens = []
        i = 0
        for idx in sorted_indices:
            sorted_X_batch[i][:] = X_batch[idx][:]
            sorted_y_batch[i, :] = y_batch[idx][:]
            sorted_y_lens.append(y_lens[i])
            sorted_tourist_loc_batch[i] = tourist_loc_batch[idx]
            sorted_mask[i, :sorted_seq_lens[i]] = 1.0
            i += 1

        return (sorted_X_batch,
                to_variable([sorted_mask,
                             sorted_tourist_loc_batch,
                             sorted_y_batch],
                            cuda=self.use_cuda),
                sorted(seq_lens, reverse=True),
                sorted_y_lens,
                max_y_len)


    def train(self, num_epochs=None):
        print("Beginning Training...")
        if num_epochs is None:
            num_epochs = self.num_epochs
        Xs, tourist_locs, landmarks, ys = self.train_data
        train_loss, train_acc = None, None
        best_valid = float('inf')
        valid_patience = 0

        to_log = time.time()
        start = time.time()
        for epoch_num in range(self.num_epochs):
            Xs, tourist_locs, ys = shuffle(Xs, tourist_locs, ys)
            total_loss, accs, total = 0.0, 0.0, 0.0
            batch_num = 0
            for jj in range(0, len(Xs), self.bsz):
                batch_num += 1
                data = self.create_batch(Xs[jj:jj + self.bsz],
                                         tourist_locs[jj:jj + self.bsz],
                                         ys[jj:jj + self.bsz])
                X_batch, (mask, t_locs_batch, y_batch), X_lengths, y_lengths, max_len = data
                res = self.model.forward(src_var=X_batch,
                                         src_lengths=X_lengths,
                                         trg_var=y_batch,
                                         trg_lengths=y_lengths,
                                         max_length=max_len,
                                         encoder_mask=mask,
                                         return_attention=True,
                                         train=True)
                total += 1
                loss = res['loss']
                total_loss += loss['loss'].cpu().data.numpy()
                self.optim.zero_grad()
                loss['loss'].backward()
                self.optim.step()
                if time.time() - to_log >= self.log_time:
                    elapsed = time.time() - start
                    print('Elapsed_time: {}, Batch: {}/{}; batch loss: {:.2f}; '.format(int(elapsed), batch_num, int(len(Xs)/self.bsz), loss['loss']))
                    to_log = time.time()
                    pred = res['preds'][0, :]
                    print('target: {}'.format(self.dictionary.decode(y_batch[0, :])))
                    print('generate: {}'.format(self.dictionary.decode(pred)))
                    # print('target: {}'.format(y_batch[0, :]))
                    # print('generate: {}'.format(pred))
                    print('\n')
            print('Epoch: {}, Loss: {}'.format(epoch_num, total_loss/total))
            valid_loss = self.eval_epoch()
            self.lr_scheduler.step(valid_loss)
            if valid_loss < best_valid:
                print('NEW BEST VALID: {}'.format(valid_loss))
                best_valid = valid_loss
                valid_patience = 0
            else:
                valid_patience += 1
                print("BEST VALID STILL GOOD AFTER {} EPOCHS".format(valid_patience))
                if valid_patience == self.valid_patience:
                    print("Finished training; saving model to {}".format(self.model_file))
                    self.save_model()
                    test_loss = self.eval_test()
                    print('Test Loss: {}'.format(test_loss))
                    return

        print('Finished {} epochs; saving anyway...'.format(self.num_epochs))
        self.save_model()
        val_loss = self.eval_epoch()
        print('Validation Loss: {}'.format(val_loss))
        test_loss = self.eval_test()
        print('Test Loss: {}'.format(test_loss))
            # train_loss = loss/total
            # train_acc = accs/total
            # print(train_loss)
            # print(train_acc)

    def eval_epoch(self):
        Xs, tourist_locs, landmarks, ys = self.valid_data
        Xs, tourist_locs, ys = shuffle(Xs, tourist_locs, ys)
        total_loss, total = 0.0, 0.0
        batch_num = 0
        for jj in range(0, len(Xs), self.bsz):
            batch_num += 1
            data = self.create_batch(Xs[jj:jj + self.bsz],
                                     tourist_locs[jj:jj + self.bsz],
                                     ys[jj:jj + self.bsz])
            X_batch, (mask, t_locs_batch, y_batch), X_lengths, y_lengths, max_len = data
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
        Xs, tourist_locs, landmarks, ys = self.test_data
        Xs, tourist_locs, ys = shuffle(Xs, tourist_locs, ys)
        total_loss, total = 0.0, 0.0
        batch_num = 0
        for jj in range(0, len(Xs), self.bsz):
            batch_num += 1
            data = self.create_batch(Xs[jj:jj + self.bsz],
                                     tourist_locs[jj:jj + self.bsz],
                                     ys[jj:jj + self.bsz])
            X_batch, (mask, t_locs_batch, y_batch), X_lengths, y_lengths, max_len = data
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

        Xs, tourist_locs, landmarks, ys = data

        if start is None:
            start = random.choice(range(len(Xs)-num_preds))

        for jj in range(start, len(Xs), num_preds):
            data = self.create_batch(Xs[jj:jj + num_preds],
                                     tourist_locs[jj:jj + num_preds],
                                     ys[jj:jj + num_preds])
            X_batch, (mask, t_locs_batch, y_batch), X_lengths, y_lengths, max_d_len = data
            if max_len is None:
                max_len = max_d_len
            res = self.model.forward(src_var=X_batch,
                                     src_lengths=X_lengths,
                                     trg_var=None,
                                     trg_lengths=None,
                                     max_length=max_len,
                                     return_attention=True)
            preds = res['preds']
            if print_preds:
                for i in range(num_preds):
                    pred = preds[i, :]
                    print('target: {}'.format(self.dictionary.decode(y_batch[i, :])))
                    print('generate: {}'.format(self.dictionary.decode(pred)))
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

        Xs, tourist_locs, landmarks, ys = data
        data = self.create_batch(Xs[idx:idx + 1],
                                 tourist_locs[idx:idx + 1],
                                 ys[idx:idx + 1])
        X_batch, (mask, t_locs_batch, y_batch), X_lengths, y_lengths, max_len = data
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
