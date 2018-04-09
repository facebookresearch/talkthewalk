import argparse
import os
import json
import torch
import torch.optim as optim
from collections import deque

from sklearn.utils import shuffle
from torch.autograd import Variable
from data_loader import Landmarks, step_aware, load_features, \
    FasttextFeatures, GoldstandardFeatures, ResnetFeatures
from dict import Dictionary, START_TOKEN, END_TOKEN, UNK_TOKEN, PAD_TOKEN
from seq2seq import Seq2Seq

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
        parser.add_argument('--cuda', action='store_true')
        parser.add_argument('--valid_patience', type=int, default=5)
        parser.add_argument('-mf', '--model-file', type=str, default='my_model')
        parser.add_argument('--resnet-features', action='store_true')
        parser.add_argument('--fasttext-features', action='store_true')
        parser.add_argument('--goldstandard-features', action='store_true')
        parser.add_argument('--condition-on-action', action='store_true')
        parser.add_argument('--mask-conv', action='store_true')
        parser.add_argument('--num-steps', type=int, default=-1)
        parser.add_argument('--softmax', choices=['landmarks', 'location'],
                            default='landmarks')
        parser.add_argument('--emb-sz', type=int, default=32)
        parser.add_argument('--hsz', type=int, default=128)
        parser.add_argument('--num-epochs', type=int, default=500)
        parser.add_argument('--batch_sz', type=int, default=64)
        parser.add_argument('--exp-name', type=str, default='test')
        parser.add_argument('--contextlen', type=int, default=-1)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--bidirectional', action='store_true')
        parser.add_argument('--attention', type=str, default='')
        parser.add_argument('--pass-hidden-state', action='store_true')
        parser.add_argument('--use-dec-state', action='store_true')
        parser.add_argument('--rnn-type', type=str, default='LSTM')
        parser.add_argument('--use-prev-word', action='store_true')
        parser.add_argument('--n-enc-layers', type=int, default=1)
        parser.add_argument('--n-dec-layers', type=int, default=1)

        parser.set_defaults(data_dir='data/',
                            goldstandard_features=True,
                            bidirectional=False,
                            pass_hidden_state=True,
                            use_dec_state=True,
                            use_prev_word=True,
                            cuda=False)
        self.args = parser.parse_args()

    def __init__(self):
        self.setup_args()
        args = self.args
        self.data_dir = args.data_dir
        self.emb_sz = args.emb_sz
        self.hsz = args.hsz
        self.num_epochs = args.num_epochs
        self.bsz = args.batch_sz
        self.contextlen = args.num_steps
        self.bidirectional = args.bidirectional
        self.attention = args.attention
        self.pass_hidden_state = args.pass_hidden_state
        self.rnn_type = args.rnn_type
        self.use_prev_word = args.use_prev_word
        self.use_dec_state = args.use_dec_state
        self.n_enc_layers = args.n_enc_layers
        self.n_dec_layers = args.n_dec_layers
        self.dropout = args.dropout
        self.use_cuda = torch.cuda.is_available() and args.cuda
        self.valid_patience = args.valid_patience
        self.model_file = args.model_file
        self.contextlen = args.contextlen if args.contextlen >= 0 else None

        self.neighborhoods = ['fidi', 'hellskitchen', 'williamsburg',
                              'uppereast', 'eastvillage']
        self.landmark_map = Landmarks(self.neighborhoods,
                                      include_empty_corners=True)
        self.dictionary = Dictionary('./data/dict.txt', 3)
        self.action_obs_dict = ActionObservationDictionary(self.landmark_map.itos, [1, 2, 3])
        print('Loading Datasets...')
        self.load_datasets()
        self.setup_feature_loaders()
        print('Building Train Data...')
        self.train_data = self.load_data(self.train_set,
                                         'train',
                                         self.feature_loaders['goldstandard'])
        print('Building Valid Data...')
        self.valid_data = self.load_data(self.valid_set,
                                         'valid',
                                         self.feature_loaders['goldstandard'])
        print('Building Test Data...')
        self.test_data = self.load_data(self.test_set,
                                        'test',
                                        self.feature_loaders['goldstandard'])
        self.max_len = max([len(seq) for seq in self.train_data[0]])
        self.model = Seq2Seq(n_lands=11,
                             n_acts=3,
                             n_words_trg=len(self.dictionary),
                             hidden_size=self.hsz,
                             emb_dim=self.emb_sz,
                             n_enc_layers=self.n_enc_layers,
                             n_dec_layers=self.n_dec_layers,
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
                             max_length=self.max_len)
        self.optim = optim.Adam(self.model.parameters())

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

    def load_data(self, dataset, dataset_name, feature_loader):
        Xs = []         # x_i = [a_1, o_1, a_2, ..., a_n, o_n] acts + obs
        tourist_locs = []
        landmarks = []
        ys = []         # y_i = msg from tourist
        dataset_path = os.path.join(self.data_dir, "{}_NLG_data/".format(dataset_name))
        if os.path.exists(dataset_path):
            data = []
            for d in ['Xs', 'tourist_locs', 'landmarks', 'ys']:
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
                            y = self.dictionary.encode(msg['text'], include_end=True)
                            ls, tourist_loc = self.landmark_map.get_landmarks_2d(
                                            neighborhood, boundaries, loc)
                            landmarks.append(ls)
                            obs_emb = feature_loader.get(neighborhood, loc)
                            act_obs_memory.append(obs_emb)

                            Xs.append(list(act_obs_memory) + [self.action_obs_dict.tok2i[END_TOKEN]])
                            ys.append(y)
                            tourist_locs.append(tourist_loc)
                            act_obs_memory.clear()
                        else:
                            loc = step_aware(act, loc, boundaries)
                            act_obs_memory.append(act)
                            if act == 2:  # went forward
                                ls, _ = self.landmark_map.get_landmarks_2d(
                                                neighborhood, boundaries, loc)
                                landmarks.append(ls)
                                obs_emb = feature_loader.get(neighborhood, loc)
                                act_obs_memory.append(obs_emb)
            os.makedirs(dataset_path)
            data = [Xs, tourist_locs, landmarks, ys]
            for i, d in enumerate(['Xs', 'tourist_locs', 'landmarks', 'ys']):
                with open(os.path.join(dataset_path, '{}.json'.format(d)), 'w') as f:
                    json.dump(data[i], f)

        return data

    def create_batch(self, Xs, tourist_locs, ys):
        batch_size = len(Xs)
        seq_lens = [len(seq) for seq in Xs]
        max_len = max(seq_lens)
        X_batch = [[0 for _ in range(max_len)] for _ in range(batch_size)]
        mask = torch.FloatTensor(batch_size, max_len).zero_()
        for i, seq in enumerate(Xs):
            for j, elem in enumerate(seq):
                X_batch[i][j] = elem
            mask[i, :len(seq)] = 1.0
        max_y_len = max(len(seq) for seq in ys)
        y_batch = torch.LongTensor(batch_size, max_y_len).fill_(self.dictionary[PAD_TOKEN])
        for i, seq in enumerate(ys):
            y_batch[i, :len(seq)] = torch.LongTensor(seq)
        # y_batch = torch.LongTensor(ys)
        tourist_loc_batch = torch.LongTensor(tourist_locs)

        # Sort batch according to length of sequence
        sorted_seq_lens, sorted_indices = torch.sort(
            torch.LongTensor(seq_lens),
            descending=True)
        sorted_X_batch = [[self.action_obs_dict.tok2i[PAD_TOKEN] for _ in range(max_len)] for _ in range(batch_size)]
        sorted_y_batch = torch.LongTensor(batch_size, max_y_len).zero_()
        sorted_tourist_loc_batch = torch.LongTensor(tourist_loc_batch.size())
        sorted_mask = torch.FloatTensor(batch_size, max_len).zero_()
        i = 0
        for idx in sorted_indices:
            sorted_X_batch[i][:] = X_batch[idx][:]
            sorted_y_batch[i, :] = y_batch[idx][:]
            sorted_tourist_loc_batch[i] = tourist_loc_batch[idx]
            sorted_mask[i, :sorted_seq_lens[i]] = 1.0
            i += 1

        return (sorted_X_batch,
                to_variable([sorted_mask,
                             sorted_tourist_loc_batch,
                             sorted_y_batch],
                            cuda=self.use_cuda),
                sorted(seq_lens, reverse=True),
                max_len)


    def train(self, num_epochs=None):
        print("Beginning Training...")
        if num_epochs is None:
            num_epochs = self.num_epochs
        Xs, tourist_locs, landmarks, ys = self.train_data

        train_loss, train_acc = None, None
        best_valid = float('inf')
        valid_patience = 0
        for epoch_num in range(self.num_epochs):
            Xs, tourist_locs, ys = shuffle(Xs, tourist_locs, ys)
            total_loss, accs, total = 0.0, 0.0, 0.0
            batch_num = 0
            for jj in range(0, len(Xs), self.bsz):
                batch_num += 1
                data = self.create_batch(Xs[jj:jj + self.bsz],
                                         tourist_locs[jj:jj + self.bsz],
                                         ys[jj:jj + self.bsz])
                X_batch, (mask, t_locs_batch, y_batch), X_lengths, max_len = data
                res = self.model.forward(src_var=X_batch,
                                         src_lengths=X_lengths,
                                         trg_var=y_batch,
                                         trg_lengths=None,
                                         max_length=max_len,
                                         return_attention=True)
                total += 1
                loss = res['loss']
                total_loss += loss['loss'].cpu().data.numpy()
                self.optim.zero_grad()
                loss['loss'].backward()
                self.optim.step()
                if batch_num % 20 == 0:
                    print('Batch: {}; batch loss: {}'.format(batch_num, loss['loss']))
            print('Epoch: {}, Loss: {}'.format(epoch_num, total_loss/(total*self.bsz)))
            valid_loss = self.eval_epoch()
            if valid_loss < best_valid:
                print('NEW BEST VALID:'.format(valid_loss))
                best_valid = valid_loss
                valid_patience = 0
            else:
                valid_patience += 1
                print("BEST VALID STILL GOOD AFTER {} EPOCHS".format(valid_patience))
                if valid_patience == self.valid_patience:
                    print("Finished training; saving model to {}".format(self.model_file))
                    torch.save(self.model.state_dict(), self.model_file)
                    return

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
            X_batch, (mask, t_locs_batch, y_batch), X_lengths, max_len = data
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

    def test_predict(self):
        Xs, tourist_locs, landmarks, ys = self.test_data
        for jj in range(0, len(Xs), self.bsz):
            data = self.create_batch(Xs[jj:jj + self.bsz],
                                     tourist_locs[jj:jj + self.bsz],
                                     ys[jj:jj + self.bsz])
            X_batch, (mask, t_locs_batch, y_batch), X_lengths, max_len = data
            res = self.model.forward(src_var=X_batch,
                                     src_lengths=X_lengths,
                                     trg_var=None,
                                     trg_lengths=None,
                                     max_length=max_len,
                                     return_attention=True)
            preds = res['preds']

            for i in range(self.bsz):
                pred = preds[i, :]
                print('target: {}'.format(self.dictionary.decode(y_batch[i, :])))
                print('generate: {}'.format(self.dictionary.decode(pred)))
                print('\n')
            break
            print(preds)

    def load_model(self, model_file):
        self.model.load_state_dict(torch.load(model_file))

if __name__ == '__main__':
    """
        TODO:
            1. figure out where to get these damn Embeddings
            2. actually implement the S2S model lol
            3. it looks like the embeddings are actually part of the model

    """
    trainer = TrainLanguageGenerator()
    trainer.train()
    # trainer.load_model('no_attention')
    # trainer.test_predict()
