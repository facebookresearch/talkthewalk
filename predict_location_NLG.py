import argparse
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from collections import deque

from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from data_loader import Landmarks, step_aware, load_features, \
    FasttextFeatures, GoldstandardFeatures, ResnetFeatures
from utils import create_logger
from dict import Dictionary, START_TOKEN, END_TOKEN, UNK_TOKEN, PAD_TOKEN
from predict_location_multiple_step import MapEmbedding2d


def get_action(msg):
    msg_to_act = {'ACTION:TURNLEFT': 1,
                  'ACTION:TURNRIGHT': 2,
                  'ACTION:FORWARD': 3}
    return msg_to_act.get(msg, None)

def to_variable(obj, cuda=True):
    if torch.is_tensor(obj):
        var = Variable(obj)
        if cuda:
            var = var.cuda()
        return var
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [to_variable(x, cuda=cuda) for x in obj]
    if isinstance(obj, dict):
        return {k: to_variable(v, cuda=cuda) for k, v in obj.items()}


class Encoder(nn.Module):

    def __init__(self, n_lands=9, n_acts=3, hidden_size=128, emb_dim=32,
                 n_layers=1, dropout=0.,
                 bidirectional=False, vocab=None, rnn_type='LSTM'):

        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.pad_idx = vocab.tok2i[PAD_TOKEN]
        self.start_idx = vocab.tok2i[START_TOKEN]
        self.end_idx = vocab.tok2i[END_TOKEN]
        self.n_acts = n_acts

        self.action_embedding = nn.Embedding(n_acts + 2, emb_dim)
        self.observation_embedding = nn.Embedding(n_lands + 2, emb_dim)
        self.dropout_layer = nn.Dropout(dropout)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(emb_dim,
                               hidden_size, batch_first=True,
                               bidirectional=bidirectional,
                               dropout=dropout, num_layers=n_layers)
        else:
            self.rnn = nn.GRU(emb_dim,
                              hidden_size, batch_first=True,
                              bidirectional=bidirectional,
                              dropout=dropout, num_layers=n_layers)

    def forward(self, x, input_lengths):
        """
            x: inputs, [bsz, T] (after applying forward embedding)
            input_lengths: real lengths of input batch
        """
        embedded = self.forward_embedding(x)
        embedded = self.dropout_layer(embedded)
        packed = pack_padded_sequence(embedded, input_lengths,
                                      batch_first=True)
        outputs, hidden = self.rnn(packed)
        outputs, output_lengths = pad_packed_sequence(outputs,
                                                      batch_first=True)

        if self.bidirectional:
            if self.rnn_type == 'LSTM':
                hidden_h = [torch.cat((hidden[0][i], hidden[0][i + 1]), dim=1)
                            for i in range(0, self.n_layers * 2, 2)]
                hidden_h = torch.stack(hidden_h, dim=0)
                hidden_c = [torch.cat((hidden[1][i], hidden[1][i + 1]), dim=1)
                            for i in range(0, self.n_layers * 2, 2)]
                hidden_c = torch.stack(hidden_c, dim=0)
                hidden = (hidden_h, hidden_c)
            else:
                hidden = [torch.cat((hidden[i], hidden[i + 1]), dim=1)
                          for i in range(0, self.n_layers * 2, 2)]
                hidden = torch.stack(hidden, dim=0)

        return outputs, hidden, embedded

    def forward_embedding(self, x):
        bsz = len(x)
        emb_size = len(x[0])
        emb_x = torch.FloatTensor(bsz, emb_size, self.emb_dim).zero_()
        for i in range(bsz):
            ex = x[i]
            for j in range(len(ex)):
                a_or_o = ex[j]
                if type(a_or_o) is list:  # list of observations
                    emb = torch.Tensor(self.emb_dim).zero_()
                    for obs in a_or_o:
                        # obs = Variable(torch.LongTensor([obs]))
                        obs = torch.LongTensor([obs])
                        emb += self.observation_embedding(obs).view(-1)
                else:  # action
                    act = Variable(torch.LongTensor([a_or_o]))
                    emb = self.action_embedding(act).data
                emb_x[i, j] = emb
        return emb_x


class Decoder(nn.Module):
    """
        Decoder with attention
    """
    def __init__(self, hidden_size=128, emb_dim=32, n_words=0, n_layers=1,
                 dropout=0.1, use_attention=False,
                 encoder_is_bidirectional=True, enc_hidden_size=128,
                 pass_hidden_state=False, vocab=None, rnn_type='LSTM',
                 ctx_dim=0, use_prev_word=True, use_dec_state=True,
                 max_length=0):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.vocab = vocab
        self.pad_idx = vocab.tok2i[PAD_TOKEN]
        self.unk_idx = vocab.tok2i[UNK_TOKEN]
        self.eos_idx = vocab.tok2i[END_TOKEN]
        self.dropout = dropout
        self.use_prev_word = use_prev_word
        self.use_dec_state = use_dec_state
        self.use_cuda = torch.cuda.is_available()
        self.pass_hidden_state = pass_hidden_state
        self.use_attention = use_attention
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        if enc_hidden_size == 0:
            enc_hidden_size = hidden_size * 2 if encoder_is_bidirectional else hidden_size
        self.embedding = nn.Embedding(n_words, emb_dim)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, hidden_size,
                              batch_first=True, dropout=dropout,
                              num_layers=n_layers)
        else:
            self.rnn = nn.LSTM(emb_dim, hidden_size,
                               batch_first=True, dropout=dropout,
                               num_layers=n_layers)

        if self.use_attention:
            self.attn = nn.Linear(emb_dim + enc_hidden_size, max_length)
            self.attn_combine = nn.Linear(max_length + hidden_size, emb_dim)

        # Projected Final Encoder State
        self.enc_to_dec_h0 = nn.Linear(enc_hidden_size, hidden_size, bias=False)
        self.enc_to_dec_c0 = nn.Linear(enc_hidden_size, hidden_size, bias=False) if rnn_type == 'LSTM' else None
        self.enc_to_dec_h1 = nn.Linear(enc_hidden_size, hidden_size, bias=False) if n_layers > 1 else None
        self.enc_to_dec_c1 = nn.Linear(enc_hidden_size, hidden_size, bias=False) if n_layers > 1 and rnn_type == 'LSTM' else None

        self.pre_output_layer = nn.Linear(enc_hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, n_words)
        self.emb_dropout = nn.Dropout(self.dropout)
        self.to_predict_dropout = nn.Dropout(self.dropout)

    def init_hidden(self, encoder_final):
        if self.pass_hidden_state:
            if self.n_layers == 1:
                if isinstance(encoder_final, tuple):  # LSTM
                    hidden = (self.enc_to_dec_h0(encoder_final[0]),
                              self.enc_to_dec_c0(encoder_final[1]))
                else:  # gru
                    hidden = self.enc_to_dec_h0(encoder_final)
            elif self.n_layers == 2:
                if isinstance(encoder_final, tuple):  # LSTM
                    h0 = self.enc_to_dec_h0(encoder_final[0][0])
                    h1 = self.enc_to_dec_h1(encoder_final[0][1])
                    h = torch.stack([h0, h1], dim=0)
                    c0 = self.enc_to_dec_c0(encoder_final[1][0])  # [B, D]
                    c1 = self.enc_to_dec_c1(encoder_final[1][1])  # [B, D]
                    c = torch.stack([c0, c1], dim=0)
                    hidden = (h, c)
                else:
                    h0 = self.enc_to_dec_h0(encoder_final[0])
                    h1 = self.enc_to_dec_h1(encoder_final[1])
                    hidden = torch.stack([h0, h1], dim=0)
            else:
                raise NotImplementedError
        else:
            if isinstance(encoder_final, tuple):
                batch_size = encoder_final[0].size(1)
                shape = [self.n_layers, batch_size, self.dim]
                hidden_h = Variable(torch.zeros(shape))
                hidden_h = hidden_h.cuda() if self.use_cuda else hidden_h
                hidden_c = Variable(torch.zeros(shape))
                hidden_c = hidden_c.cuda() if self.use_cuda else hidden_c
                hidden = (hidden_h, hidden_c)

            else:  # gru
                batch_size = encoder_final.size(1)
                shape = [self.n_layers, batch_size, self.dim]
                hidden = Variable(torch.zeros(shape))
                hidden = hidden.cuda() if self.use_cuda else hidden

        return hidden

    def forward(self, encoder_outputs=None, encoder_final=None,
                encoder_mask=None, max_length=0, trg_var=None,
                return_attention=False,
                return_states=True):

        """
        Forward decoding pass.

        Args:
            encoder_outputs: encoder hidden states (B, T, 2D)
            encoder_final: encoder final state (forward, backward concatenated)
            encoder_mask: mask of actual encoder positions (B, T)
            max_length: maximum number of decoding steps (for prediction)
            trg_var: variable with target indexes for teacher forcing
            return_attention: return the attention scores
            return_states: return decoder states
        """

        bsz = encoder_outputs.size(0)  # B

        # start of sequence = embedding all 0s
        embedded = Variable(torch.zeros((bsz, 1, self.emb_dim)))
        embedded = embedded.cuda() if self.use_cuda else embedded
        hidden = self.init_hidden(encoder_final)  # (n_layers, B, hsz)



        all_predictions = []
        all_log_probs = []
        all_attention_scores = []
        decoder_states = []
        masks = []

        # mask everything after </s> was generated
        mask = Variable(torch.ones([bsz, 1]).byte())
        mask = mask.cuda() if self.use_cuda else mask

        for i in range(max_length):
            masks.append(mask)
            if self.use_attention:
                embedded = embedded.view(1, bsz, -1)
                if self.rnn_type == 'LSTM':
                    hidden_attn = hidden[0][0]
                else:
                    hidden_attn = hidden[0]
                attn_weights = F.softmax(
                    self.attn(torch.cat((embedded[0], hidden_attn), 1)), dim=1)
                if return_attention:
                    all_attention_scores.append(attn_weights)
                attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.transpose(1, 2).squeeze(1))
                output = torch.cat((embedded[0], attn_applied[0]), 1)
                output = self.attn_combine(output).unsqueeze(0)
                output = F.relu(output)
            else:
                output = embedded
            _, hidden = self.rnn(output, hidden)  # hidden (n_layers, B, hsz)

            # predict from (top) RNN hidden state directly
            to_predict = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]  # [B, V]

            if return_states:
                decoder_states.append(to_predict.unsqueeze(0))
            if self.use_attention:
                if self.use_prev_word and self.use_dec_state:
                    to_predict = torch.cat(
                                    (to_predict.unsqueeze(1),
                                        embedded,
                                        output),
                                    2)
                elif self.use_prev_word and not self.use_dec_state:
                    to_predict = torch.cat((embedded, output), 2)
                elif not self.use_prev_word and not self.use_dec_state:
                    to_predict = output

            to_predict = self.pre_output_layer(to_predict)       # (B, 1, D)
            logits = self.output_layer(to_predict.unsqueeze(1))  # (B, 1, V)

            log_probs = F.log_softmax(logits, 2)
            all_log_probs.append(log_probs)
            predictions = log_probs.max(2)[1]  # [B, 1]
            all_predictions.append(predictions)

            mask = (predictions != self.eos_idx) * mask

            if trg_var is not None:  # teacher forcing, feed true targets to next step
                targets_this_iter = trg_var[:, i, None]       # (B, 1)
                embedded = self.embedding(targets_this_iter)  # (B, 1, E)
                embedded = self.emb_dropout(embedded)
            else:  # feed current predictions to next step
                embedded = self.embedding(predictions)   # (B, 1, E)
                embedded = self.emb_dropout(embedded)

        all_predictions = torch.cat(all_predictions, 1)  # (B, T)
        all_log_probs = torch.cat(all_log_probs, 1)      # (B, T, V)
        mask = torch.cat(masks, 1)

        if return_states:
            decoder_states = torch.cat(decoder_states, 0)    # (T, B, D)
            decoder_states = decoder_states.transpose(0, 1)  # (T, B, D) -> (B, T, D)
        else:
            decoder_states = None
        if return_attention and self.use_attention:
            all_attention_scores = torch.cat(all_attention_scores, 1)      # (B, T', T)
        else:
            all_attention_scores = None

        mask = torch.cat(masks, 1)

        return {'preds': all_predictions,
                'log_probs': all_log_probs,
                'att_scores': all_attention_scores,
                'states': decoder_states,
                'mask': mask}


class Seq2Seq(nn.Module):
    """Seq2Seq enc/dec (NO RL CURRENTLY)"""
    def __init__(self, n_lands=9, n_acts=3, n_words_trg=0, hidden_size=0,
                 emb_dim=0,
                 n_enc_layers=1, n_dec_layers=1, dropout=0., word_dropout=0.,
                 bidirectional=False, use_attention=False,
                 pass_hidden_state=True, vocab_src=None, vocab_trg=None,
                 rnn_type=None,
                 ctx_dim=0, use_prev_word=True, use_dec_state=True, max_length=0,
                 **kwargs):

        super(Seq2Seq, self).__init__()
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim

        self.n_landmarks = n_lands
        self.n_acts = n_acts
        self.n_words_trg = n_words_trg

        self.dropout = dropout

        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg

        self.rnn_type = rnn_type

        self.pass_hidden_state = pass_hidden_state

        self.src_tagger = None
        self.trg_tagger = None

        self.src_pad_idx = vocab_src.tok2i[PAD_TOKEN]
        self.src_unk_idx = vocab_src.tok2i[UNK_TOKEN]
        self.trg_pad_idx = vocab_trg.tok2i[PAD_TOKEN]
        self.trg_unk_idx = vocab_trg.tok2i[UNK_TOKEN]

        self.criterion = nn.NLLLoss(reduce=False, size_average=False,
                                    ignore_index=self.trg_pad_idx)

        self.encoder = Encoder(n_lands, n_acts, hidden_size, emb_dim,
                               n_enc_layers, dropout, bidirectional, vocab_src,
                               rnn_type)
        self.decoder = Decoder(hidden_size, emb_dim, n_words_trg, n_dec_layers,
                               dropout, use_attention, bidirectional,
                               hidden_size, pass_hidden_state, vocab_trg,
                               rnn_type, ctx_dim, use_prev_word,
                               use_dec_state, max_length)


    def forward(self, src_var=None, src_lengths=None, trg_var=None,
                trg_lengths=None, max_length=0,
                return_attention=False,):
        """

        Args:
            src_var:
            src_lengths:
            trg_var:
            trg_lengths:
            max_length: required when trg_var is not given
            return_attention: return attention scores

        Returns:

        """
        trg_max_length = trg_var.size(1) if trg_var is not None else max_length

        input_mask = (src_var != self.src_pad_idx)
        encoder_outputs,\
            encoder_final,\
            embedded = self.encoder(src_var, src_lengths)

        result = self.decoder(
            encoder_final=encoder_final, encoder_outputs=encoder_outputs,
            encoder_mask=input_mask, max_length=trg_max_length,
            trg_var=trg_var, return_attention=return_attention,
            )
        if trg_var is not None:
            result['loss'] = self.get_loss(result, trg_var)

        return result

    def get_loss(self, result=None, trg_var=None):

        # all masks are ByteTensors here; convert to float() when multiplying them with non-ByteTensors
        trg_mask = (trg_var != self.trg_pad_idx)
        mask = result['mask']  # this is false for items after </s> was predicted
        mask = mask * trg_mask       # mask if predicted sequence longer than target sequence
        padding_mask = 1-mask  # this is positive after </s> predicted

        log_probs = result['log_probs']
        batch_size = log_probs.size(0)
        time = log_probs.size(1)
        voc_size = log_probs.size(2)

        log_probs_2d = log_probs.view(-1, voc_size)
        ce_loss = self.criterion(log_probs_2d, trg_var.view(-1))
        ce_loss = ce_loss.view([batch_size, time])
        ce_loss = ce_loss.masked_fill(padding_mask, 0.)
        ce_loss = ce_loss.sum() / batch_size

        return {'loss': ce_loss}


class TrainLanguageGenerator(object):
    """class for training the language generator. Provides a trainloop"""
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
        self.use_cuda = torch.cuda.is_available()
        self.valid_patience = args.valid_patience
        self.model_file = args.model_file

        self.neighborhoods = ['fidi', 'hellskitchen', 'williamsburg',
                              'uppereast', 'eastvillage']
        self.contextlen = args.contextlen if args.contextlen >= 0 else None
        self.landmark_map = Landmarks(self.neighborhoods,
                                      include_empty_corners=True)
        self.dictionary = Dictionary('./data/dict.txt', 3)
        self.load_datasets()
        self.setup_feature_loaders()
        self.train_data = self.load_data(self.train_set,
                                         'train',
                                         self.feature_loaders['goldstandard'])
        self.valid_data = self.load_data(self.valid_set,
                                         'valid',
                                         self.feature_loaders['goldstandard'])
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
                             use_attention=self.attention,
                             pass_hidden_state=self.pass_hidden_state,
                             vocab_src=self.dictionary,
                             vocab_trg=self.dictionary,
                             rnn_type=self.rnn_type,
                             ctx_dim=0,
                             use_prev_word=self.use_prev_word,
                             use_dec_state=True,
                             max_length=self.max_len)
        self.optim = optim.Adam(self.model.parameters())

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
        parser.add_argument('--attention', action='store_true')
        parser.add_argument('--pass-hidden-state', action='store_true')
        parser.add_argument('--use-dec-state', action='store_true')
        parser.add_argument('--rnn-type', type=str, default='LSTM')
        parser.add_argument('--use-prev-word', action='store_true')
        parser.add_argument('--n-enc-layers', type=int, default=1)
        parser.add_argument('--n-dec-layers', type=int, default=1)

        parser.set_defaults(data_dir='data/',
                            goldstandard_features=True,
                            bidirectional=False,
                            attention=False,
                            pass_hidden_state=True,
                            use_dec_state=True,
                            use_prev_word=True)
        self.args = parser.parse_args()

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
                            y = self.dictionary.encode(msg['text'])
                            ls, tourist_loc = self.landmark_map.get_landmarks_2d(
                                            neighborhood, boundaries, loc)
                            landmarks.append(ls)
                            obs_emb = feature_loader.get(neighborhood, loc)
                            act_obs_memory.append(obs_emb)

                            Xs.append(list(act_obs_memory))
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
        #
        # X_batch = torch.LongTensor(batch_size, max_len).zero_()
        # X_batch = np.zeros((batch_size, max_len))
        X_batch = [[0 for _ in range(max_len)] for _ in range(batch_size)]
        mask = torch.FloatTensor(batch_size, max_len).zero_()
        for i, seq in enumerate(Xs):
            for j, elem in enumerate(seq):
                X_batch[i][j] = elem
            mask[i, :len(seq)] = 1.0
        #
        # max_landmarks_per_coord = max([max([max([len(y) for y in x]) for x in l]) for l in landmarks])
        # landmark_batch = torch.LongTensor(batch_size, 4, 4, max_landmarks_per_coord).zero_()
        #
        # for i, ls in enumerate(landmarks):
        #     for j in range(4):
        #         for k in range(4):
        #             landmark_batch[i, j, k, :len(landmarks[i][j][k])] = torch.LongTensor(landmarks[i][j][k])
        #
        max_y_len = max([len(seq) for seq in ys])
        y_batch = torch.LongTensor(batch_size, max_y_len).zero_()
        for i, seq in enumerate(ys):
            y_batch[i, :len(seq)] = torch.LongTensor(seq)
        # y_batch = torch.LongTensor(ys)
        tourist_loc_batch = torch.LongTensor(tourist_locs)

        #Sort batch according to length of sequence
        sorted_seq_lens, sorted_indices = torch.sort(torch.LongTensor(seq_lens), descending=True)
        sorted_X_batch = [[0 for _ in range(max_len)] for _ in range(batch_size)]
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

        return sorted_X_batch, to_variable([sorted_mask, sorted_tourist_loc_batch,
                            sorted_y_batch], cuda=self.use_cuda), sorted(seq_lens, reverse=True), max_len


    def train(self, num_epochs=None):
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
    # trainer.load_model('temp_model')
    # trainer.test_predict()
