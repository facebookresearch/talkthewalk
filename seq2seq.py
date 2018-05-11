import random
import math
import operator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from dict import START_TOKEN, END_TOKEN, PAD_TOKEN


class BahdanauAttention(nn.Module):
    """
    Computes Bahdanau attention between a memory (e.g. encoder states)
    and a query (e.g. a decoder state)
    """

    def __init__(self, query_dim=0, memory_dim=0):
        """
        Initialize the attention mechanism.

        Args:
            query_dim:  dimensionality of the query
            memory_dim: dimensionality of the memory (e.g. encoder states)
        """
        super(BahdanauAttention, self).__init__()
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.memory_layer = nn.Linear(memory_dim, query_dim)
        self.query_layer = nn.Linear(query_dim, query_dim)
        self.energy_layer = nn.Linear(query_dim, 1)

    def project_memory(self, memory):
        """
        Simply applies a learned transformation over the given memory tensor
        Args:
            memory:

        Returns:
            projected memory

        """
        return self.memory_layer(memory)

    def forward(self, query=None, projected_memory=None, mask=None):
        proj_query = self.query_layer(query)            # (1, B, D)
        proj_query = proj_query.transpose(0, 1)         # (B, 1, D)

        # this broadcasts the query over the projected memory
        energy = F.tanh(proj_query + projected_memory)  # (B, T, D)
        energy = self.energy_layer(energy).squeeze(2)   # (B, T)

        # mask illegal attention values
        pad_mask = (mask == 0)

        energy = energy.masked_fill(pad_mask, -1e3)  # FIXME would like -inf here
        alpha = F.softmax(energy, 1)    # (B, T)
        alpha = alpha.unsqueeze(1)      # (B, 1, T)
        return alpha


class Encoder(nn.Module):

    def __init__(self, args, n_lands=10, n_acts=3, vocab=None):

        super(Encoder, self).__init__()
        self.hidden_size = args.hsz
        self.emb_dim = args.enc_emb_sz
        self.resnet_dim = args.resnet_dim
        self.dropout = args.dropout
        self.resnet_proj_dim = args.resnet_proj_dim
        self.bidirectional = args.bidirectional
        self.rnn_type = args.rnn_type
        self.pad_idx = vocab.tok2i[PAD_TOKEN]
        self.n_acts = n_acts
        self.use_cuda = torch.cuda.is_available() and args.use_cuda
        self.n_layers = args.n_layers
        if not args.orientation_aware:
            self.action_embedding = nn.Embedding(n_acts + 1, self.emb_dim,
                                                 padding_idx=self.pad_idx)
            self.observation_embedding = nn.Embedding(n_lands + 1, self.emb_dim,
                                                      padding_idx=self.pad_idx)
        else:
            self.action_embedding = nn.Embedding(n_acts + n_lands + 4,
                                                 self.emb_dim,
                                                 padding_idx=self.pad_idx)
            self.observation_embedding = nn.Embedding(n_lands + n_acts + 4,
                                                      self.emb_dim,
                                                      padding_idx=self.pad_idx)
        self.use_resnet = args.resnet_features
        if self.use_resnet:
            self.resnet_linear = nn.Linear(self.resnet_dim, self.resnet_proj_dim)
        else:
            self.resnet_proj_dim = 0

        self.dropout_layer = nn.Dropout(args.dropout)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.emb_dim + self.resnet_proj_dim, #concatenating resnet and gold standard
                               self.hidden_size, batch_first=True,
                               bidirectional=self.bidirectional,
                               dropout=self.dropout, num_layers=self.n_layers)
        else:
            self.rnn = nn.GRU(self.emb_dim + self.resnet_proj_dim,
                              self.hidden_size, batch_first=True,
                              bidirectional=self.bidirectional,
                              dropout=self.dropout, num_layers=self.n_layers)

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
        if self.use_resnet:
            res_feats = torch.FloatTensor(bsz, emb_size, self.resnet_proj_dim).zero_()
        if self.use_cuda:
            emb_x = emb_x.cuda()
            if self.use_resnet:
                res_feats = res_feats.cuda()
        for i in range(bsz):
            ex = x[i]
            for j in range(len(ex)):
                a_or_o = ex[j]
                res_feat = torch.Tensor(self.resnet_proj_dim).zero_()
                if self.use_cuda:
                    res_feat = res_feat.cuda()
                if type(a_or_o) is dict:  # list of observations
                    gold_emb = torch.Tensor(self.emb_dim).zero_()
                    if self.use_cuda:
                        gold_emb = gold_emb.cuda()
                    for k, feats in a_or_o.items():
                        for obs in feats:
                            if type(obs) is list: #resnet features
                                obs_tens = torch.FloatTensor(obs)
                                if self.use_cuda:
                                    obs_tens = obs_tens.cuda()
                                res_feat += self.resnet_linear(obs_tens)
                            else:
                                obs = [obs]
                                obs_tens = torch.LongTensor(obs)
                                if self.use_cuda:
                                    obs_tens = obs_tens.cuda()
                                gold_emb += self.observation_embedding(obs_tens).view(-1)
                else:  # action
                    act = torch.LongTensor([a_or_o])
                    if self.use_cuda:
                        act = act.cuda()
                    gold_emb = self.action_embedding(act)

                emb_x[i, j] = gold_emb
                if self.use_resnet:
                    res_feats[i, j] = res_feat
        if self.use_resnet:
            return torch.cat((emb_x, res_feats), 2)
        else:
            return emb_x


class Decoder(nn.Module):
    """
        Decoder with attention
    """
    def __init__(self, args, n_words=0,
                 vocab=None,
                 max_length=0):

        super(Decoder, self).__init__()

        self.hidden_size = args.hsz
        self.emb_dim = args.dec_emb_sz
        self.vocab = vocab
        self.n_words = n_words
        self.pad_idx = vocab.tok2i[PAD_TOKEN]
        self.eos_idx = vocab.tok2i[END_TOKEN]
        self.dropout = args.dropout
        self.use_prev_word = args.use_prev_word
        self.use_dec_state = args.use_dec_state
        self.use_cuda = torch.cuda.is_available() and args.use_cuda
        self.sample_tokens = args.sample_tokens
        self.pass_hidden_state = args.pass_hidden_state
        self.attn_type = args.attention
        self.use_attention = self.attn_type != 'none'
        self.rnn_type = args.rnn_type
        self.n_layers = args.n_layers
        self.beam_width = args.beam_width
        if args.bidirectional:
            args.hsz *= 2
        if args.hsz == 0:
            args.hsz = self.hidden_size * 2 if args.bidirectional else self.hidden_size
        self.embedding = nn.Embedding(n_words, self.emb_dim, padding_idx=self.pad_idx)
        rnn_dim = self.emb_dim if not self.use_attention else self.emb_dim + args.hsz
        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(rnn_dim, self.hidden_size,
                              batch_first=True, dropout= self.dropout,
                              num_layers=self.n_layers)
        else:
            self.rnn = nn.LSTM(rnn_dim, self.hidden_size,
                               batch_first=True, dropout=self.dropout,
                               num_layers=self.n_layers)

        if self.use_attention:
            if self.attn_type == 'Bahdanau':
                self.attention = BahdanauAttention(self.hidden_size, args.hsz)
            else:
                self.attn = nn.Linear(self.emb_dim + args.hsz, max_length)
                self.attn_combine = nn.Linear(max_length + self.hidden_size, self.emb_dim)

        # Projected Final Encoder State
        self.enc_to_dec_h0 = nn.Linear(args.hsz, self.hidden_size, bias=False)
        self.enc_to_dec_c0 = nn.Linear(args.hsz, self.hidden_size, bias=False) if self.rnn_type == 'LSTM' else None
        self.enc_to_dec_h1 = nn.Linear(args.hsz, self.hidden_size, bias=False) if self.n_layers > 1 else None
        self.enc_to_dec_c1 = nn.Linear(args.hsz, self.hidden_size, bias=False) if self.n_layers > 1 and self.rnn_type == 'LSTM' else None

        pre_output_input_dim = args.hsz
        if self.use_prev_word:
            pre_output_input_dim += self.emb_dim
        if self.use_dec_state:
            pre_output_input_dim += self.hidden_size

        self.pre_output_layer = nn.Linear(pre_output_input_dim, self.hidden_size)
        # output layer from context vector and current decoder state to n_words
        self.output_layer = nn.Linear(self.hidden_size, n_words)
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
                shape = [self.n_layers, batch_size, self.hidden_size]
                hidden_h = torch.zeros(shape)
                hidden_h = hidden_h.cuda() if self.use_cuda else hidden_h
                hidden_c = torch.zeros(shape)
                hidden_c = hidden_c.cuda() if self.use_cuda else hidden_c
                hidden = (hidden_h, hidden_c)

            else:  # gru
                batch_size = encoder_final.size(1)
                shape = [self.n_layers, batch_size, self.hidden_size]
                hidden = torch.zeros(shape)
                hidden = hidden.cuda() if self.use_cuda else hidden

        return hidden

    def forward(self, encoder_outputs=None, encoder_final=None,
                encoder_mask=None, max_length=0, trg_var=None,
                return_attention=False,
                return_states=False,
                beam_search=False):

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
        if not beam_search:
            return self.forward_sample(encoder_outputs=encoder_outputs,
                                       encoder_final=encoder_final,
                                       encoder_mask=encoder_mask,
                                       max_length=max_length,
                                       trg_var=trg_var,
                                       return_attention=return_attention,
                                       return_states=return_states)
        else:
            return self.forward_beam(encoder_outputs=encoder_outputs,
                                     encoder_final=encoder_final,
                                     encoder_mask=encoder_mask,
                                     max_length=max_length,
                                     trg_var=trg_var,
                                     return_attention=return_attention,
                                     return_states=return_states,
                                     width=self.beam_width)

    def forward_sample(self, encoder_outputs=None, encoder_final=None,
                       encoder_mask=None, max_length=0, trg_var=None,
                       return_attention=False,
                       return_states=True):

        """
        Forward decoding pass, using greedy (or sampled) decoding.

        Args:
            encoder_outputs: encoder hidden states (B, T, 2D)
            encoder_final: encoder final state (forward, backward concatenated)
            encoder_mask: mask of actual encoder positions (B, T)
            max_length: maximum number of decoding steps (for prediction)
            trg_var: variable with target indexes for teacher forcing
            return_attention: return the attention scores
            return_states: return decoder states
        """
        teacher_force = random.random() < 0.5
        bsz = encoder_outputs.size(0)  # B
        if self.attn_type == 'Bahdanau':
            projected_memory = self.attention.project_memory(encoder_outputs)
            attention_values = encoder_outputs
        # start of sequence = embedding all 0s
        embedded = torch.LongTensor(bsz, 1).fill_(self.vocab.tok2i[START_TOKEN])
        embedded = embedded.cuda() if self.use_cuda else embedded
        embedded = self.embedding(embedded)
        hidden = self.init_hidden(encoder_final)  # (n_layers, B, hsz)
        all_predictions = []
        all_log_probs = []
        all_attention_scores = []
        decoder_states = []
        masks = []

        # mask everything after </s> was generated
        mask = torch.ones([bsz, 1]).byte()
        mask = mask.cuda() if self.use_cuda else mask

        for i in range(max_length):
            masks.append(mask)
            if self.use_attention:
                if self.attn_type == 'Bahdanau':
                    query = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]  # [B, D]
                    query = query.unsqueeze(0)  # [1, B, D]
                    alpha = self.attention(query=query,
                                           projected_memory=projected_memory,
                                           mask=encoder_mask)  # (B, 1, T)
                    if return_attention:
                        all_attention_scores.append(alpha)
                    context = alpha.bmm(attention_values)  # (B, 1, D)
                    rnn_input = torch.cat((embedded, context), 2) # (B, 1, D+emb_dim)
                else:
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
                    rnn_input = F.relu(output)
            else:
                rnn_input = embedded
            dec_out, hidden = self.rnn(rnn_input, hidden)  # hidden (n_layers, B, hsz)

            # predict from (top) RNN hidden state directly
            current_state = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]  # [B, V]
            # This was wrong: context = context if self.attn_type == 'Bahdanau' else dec_out
            context = context if self.attn_type == 'Bahdanau' else dec_out
            if return_states:
                decoder_states.append(current_state.unsqueeze(0))
            if self.use_prev_word and self.use_dec_state:
                to_predict = torch.cat(
                                (current_state.unsqueeze(1),
                                    embedded,
                                    context),
                                2)
            elif self.use_prev_word and not self.use_dec_state:
                to_predict = torch.cat((embedded, context), 2)
            elif not self.use_prev_word and self.use_dec_state:
                to_predict = torch.cat((current_state.unsqueeze(1), context), 2)
            elif not self.use_prev_word and not self.use_dec_state:
                to_predict = context
            to_predict = self.pre_output_layer(to_predict)       # (B, 1, D)
            logits = self.output_layer(to_predict)  # (B, 1, V)
            log_probs = F.log_softmax(logits, 2)
            all_log_probs.append(log_probs)

            if self.sample_tokens:
                predictions = torch.LongTensor(bsz, 1).fill_(self.pad_idx)
                probs = F.softmax(logits, 2)
                for j in range(bsz):
                    probs_j = probs[j][0].tolist()
                    under_1 = (1-sum(probs_j))
                    probs_j[0] += under_1
                    predictions[j] = int(np.random.choice(range(self.n_words), p=probs_j))
                if self.use_cuda:
                    predictions = predictions.cuda()
            else:
                predictions = log_probs.max(2)[1]  # [B, 1]
            all_predictions.append(predictions)

            mask = (predictions != self.eos_idx) * mask

            if trg_var is not None and teacher_force:  # teacher forcing, feed true targets to next step
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
                'mask': mask,
                'teacher_force': teacher_force}

    def forward_beam(self, encoder_outputs=None, encoder_final=None,
                     encoder_mask=None, max_length=0, trg_var=None,
                     return_attention=False,
                     return_states=True,
                     width=4,
                     norm_pow=1.0):

        """
        Forward decoding pass, with beam search.

        Args:
            encoder_outputs: encoder hidden states (B, T, 2D)
            encoder_final: encoder final state (forward, backward concatenated)
            encoder_mask: mask of actual encoder positions (B, T)
            max_length: maximum number of decoding steps (for prediction)
            trg_var: variable with target indexes for teacher forcing
            return_attention: return the attention scores
            return_states: return decoder states
            width: width of the beam search
            norm_pow: norm power for the beam search
        """
        if self.attn_type == 'Bahdanau':
            projected_memory = self.attention.project_memory(encoder_outputs)
            attention_values = encoder_outputs

        teacher_force = random.random() < 0.5
        batch_size = encoder_outputs.size(0)  # B
        voc_size = len(self.vocab)
        live = [[(0.0, [(self.vocab.tok2i[START_TOKEN], 0)], 0)] for _ in range(batch_size)]
        dead = [[] for _ in range(batch_size)]
        num_dead = [0 for _ in range(batch_size)]

        embedded = torch.LongTensor(batch_size, 1).fill_(
                                                self.vocab.tok2i[START_TOKEN])
        embedded = embedded.cuda() if self.use_cuda else embedded
        embedded = self.embedding(embedded)

        hidden = self.init_hidden(encoder_final)  # (n_layers, B, hsz)
        all_predictions = []
        all_log_probs = []
        all_attention_scores = []
        decoder_states = []
        masks = []

        # mask everything after </s> was generated
        mask = torch.ones([batch_size, 1]).byte()
        mask = mask.cuda() if self.use_cuda else mask

        for tidx in range(max_length):
            masks.append(mask)
            if self.use_attention:
                if self.attn_type == 'Bahdanau':
                    query = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]  # [B, D]
                    query = query.unsqueeze(0)  # [1, B, D]
                    alpha = self.attention(query=query,
                                           projected_memory=projected_memory,
                                           mask=encoder_mask)  # (B, 1, T)
                    if return_attention:
                        all_attention_scores.append(alpha)
                    context = alpha.bmm(attention_values)  # (B, 1, D)
                    rnn_input = torch.cat((embedded, context), 2) # (B, 1, D+emb_dim)
                else:
                    embedded = embedded.view(1, batch_size, -1)
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
                    rnn_input = F.relu(output)
            else:
                rnn_input = embedded
            dec_out, hidden = self.rnn(rnn_input, hidden)  # hidden (n_layers, B, hsz)

            # predict from (top) RNN hidden state directly
            current_state = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]  # [B, V]
            context = context if self.attn_type == 'Bahdanau' else dec_out
            if return_states:
                decoder_states.append(current_state.unsqueeze(0))
            if self.use_prev_word and self.use_dec_state:
                to_predict = torch.cat(
                                (current_state.unsqueeze(1),
                                    embedded,
                                    context),
                                2)
            elif self.use_prev_word and not self.use_dec_state:
                to_predict = torch.cat((embedded, context), 2)
            elif not self.use_prev_word and self.use_dec_state:
                to_predict = torch.cat((current_state.unsqueeze(1), context), 2)
            elif not self.use_prev_word and not self.use_dec_state:
                to_predict = context
            to_predict = self.pre_output_layer(to_predict)       # (B, 1, D)
            logits = self.output_layer(to_predict)  # (B, 1, V)

            cur_prob = F.log_softmax(logits, 2).view(batch_size, -1, voc_size) # (B, width, V)
            all_log_probs.append(cur_prob)

            pre_prob = torch.FloatTensor([[x[0] for x in ee ] for ee in live]).view(batch_size, -1, 1) # (B, width, 1)
            pre_prob = pre_prob.cuda() if self.use_cuda else pre_prob
            total_prob = cur_prob + pre_prob # (B, width, V)
            total_prob = total_prob.view(batch_size, -1)

            topivs, topi_s = total_prob.topk(width, dim=1)
            topv_s = cur_prob.view(batch_size, -1).gather(1, topi_s)

            new_live = [[] for ii in range(batch_size)]
            for bidx in range(batch_size):
                num_live = width - num_dead[bidx]
                if num_live > 0:
                    tis = topi_s[bidx][:num_live].view(-1).tolist()
                    tvs = topv_s[bidx][:num_live].view(-1).tolist()
                    for eidx, (topi, topv) in enumerate(zip(tis, tvs)): # NOTE max width times
                        #TODO: USE THIS EIDX FOR THE PROBABILITY ROWs
                        if topi % voc_size == self.vocab.tok2i[END_TOKEN]:
                            dead[bidx].append((live[bidx][topi//voc_size][0] + topv,
                                               live[bidx][topi//voc_size][1] + [(topi % voc_size, eidx)],
                                               topi))
                            num_dead[bidx] += 1
                        else:
                            new_live[bidx].append((live[bidx][topi//voc_size][0] + topv,
                                                   live[bidx][topi//voc_size][1] + [(topi % voc_size, eidx)],
                                                   topi))

                while len(new_live[bidx]) < width:
                    new_live[bidx].append((-99999999,
                                           [0],
                                           0))
            live = new_live

            if num_dead == [width for ii in range(batch_size)]:
                break

            if trg_var is not None and teacher_force:  # teacher forcing, feed true targets to next step
                targets_this_iter = trg_var[:, tidx, None]    # (B, width)
                preds = torch.LongTensor([[x[2] % voc_size for x in ee] for ee in live])
                preds = preds.cuda() if self.use_cuda else preds
                for i in range(preds.size(0)):
                    trg_idx = targets_this_iter[i]
                    for j in range(preds[i].size(0)):
                        preds[i, j] = trg_idx
                embedded = self.embedding(preds.view(-1))  # (B * width, E)
                embedded = self.emb_dropout(embedded)
                embedded = embedded.unsqueeze(1)
            else:  # feed current predictions to next step
                preds = torch.LongTensor([[x[2] % voc_size for x in ee] for ee in live]).view(-1)
                embedded = preds.cuda() if self.use_cuda else preds
                embedded = self.embedding(embedded)   # (B * width, E)
                embedded = self.emb_dropout(embedded)
                embedded = embedded.unsqueeze(1)

            bb = 1 if tidx == 0 else width
            in_width_idx = torch.LongTensor([[x[2]//voc_size + bbidx * bb for x in ee] for bbidx, ee in enumerate(live)])
            in_width_idx = in_width_idx.cuda() if self.use_cuda else in_width_idx
            if self.rnn_type == 'LSTM':
                context = hidden[1]
                hidden = hidden[0]
            hidden = hidden.index_select( 1, in_width_idx.view(-1)).view(self.n_layers, -1, self.hidden_size)
            if tidx == 0 and self.attn_type == 'Bahdanau':
                projected_memory = projected_memory.repeat(width, 1, 1)
                encoder_mask = encoder_mask.repeat(width, 1)
                attention_values = attention_values.repeat(width, 1, 1)
            if self.rnn_type == 'LSTM':
                context = context.index_select( 1, in_width_idx.view(-1)).view(self.n_layers, -1, self.hidden_size)
                hidden = (hidden, context)

        for bidx in range(batch_size):
            if num_dead[bidx] < width:
                for didx in range(width - num_dead[bidx]):
                    (a, b, c) = live[bidx][didx]
                    dead[bidx].append((a, b, c))

        dead_ = [ [ ( float(a) / math.pow(len(b), norm_pow) , b, c) for (a,b,c) in ee] for ee in dead]
        ans = []
        compiled_log_probs = []
        seq_log_probs = []
        for bidx in range(len(dead_)):
            log_probs = []
            dd_ = dead_[bidx]
            probs = [x[0] for x in dd_]
            max_idx, _ = max(enumerate(probs), key=lambda x: x[1])
            max_dd = dd_[max_idx]
            for tidx, (tok, eidx) in enumerate(max_dd[1]):
                if tidx == len(max_dd[1]) - 1:
                    continue
                log_probs.append(all_log_probs[tidx][bidx][eidx])
            dd = sorted(dd_, key=lambda x: x[0], reverse=True)
            ans.append([x[0] for x in dd[0][1][1:-1]])
            seq_log_probs.append(dd[0][0])
            compiled_log_probs.append(log_probs)

        all_predictions = torch.LongTensor(batch_size, max_length).fill_(self.vocab.tok2i[END_TOKEN]) # (B, T)
        for i in range(len(ans)):
            pred = ans[i]
            all_predictions[i, :len(pred)] = torch.LongTensor(pred)
        all_predictions = all_predictions.cuda() if self.use_cuda else all_predictions

        all_log_probs = seq_log_probs
        mask = torch.cat(masks, 1)

        if return_states:
            decoder_states = torch.cat(decoder_states, 0)    # (T, B, D)
            decoder_states = decoder_states.transpose(0, 1)  # (T, B, D) -> (B, T, D)
        else:
            decoder_states = None
        all_attention_scores = None

        mask = torch.cat(masks, 1)
        return {'preds': all_predictions,
                'log_probs': all_log_probs,
                'att_scores': all_attention_scores,
                'states': decoder_states,
                'mask': mask,
                'teacher_force': teacher_force}


class Seq2Seq(nn.Module):
    """Seq2Seq enc/dec (NO RL CURRENTLY)"""
    def __init__(self, args, n_lands=9, n_acts=3, n_words_trg=0,
                 vocab_src=None, vocab_trg=None,
                 max_length=0,
                 **kwargs):

        super(Seq2Seq, self).__init__()
        self.fill_padding_mask = args.fill_padding_mask
        self.use_cuda = torch.cuda.is_available() and args.use_cuda

        self.src_pad_idx = vocab_src.tok2i[PAD_TOKEN]
        self.trg_pad_idx = vocab_trg.tok2i[PAD_TOKEN]

        self.criterion = nn.NLLLoss(reduce=False, size_average=False,
                                    ignore_index=self.trg_pad_idx)
        self.encoder = Encoder(args,
                               n_lands=n_lands,
                               n_acts=n_acts,
                               vocab=vocab_src)
        self.decoder = Decoder(args,
                               n_words=n_words_trg,
                               vocab=vocab_trg,
                               max_length=max_length)

    def compute_input_mask(self, src_var):
        mask = torch.LongTensor(len(src_var), len(src_var[0])).fill_(self.src_pad_idx)
        for i in range(len(src_var)):
            for j in range(len(src_var[i])):
                mask[i, j] = 1 if src_var[i][j] != self.src_pad_idx else 0
        return mask

    def forward(self, src_var=None, src_lengths=None, trg_var=None,
                trg_lengths=None, max_length=0, encoder_mask=None,
                return_attention=False, train=False):
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

        # input_mask = (src_var != self.src_pad_idx)
        input_mask = encoder_mask
        if input_mask is None:
            input_mask = self.compute_input_mask(src_var)
        if self.use_cuda:
            input_mask = input_mask.cuda()

        encoder_outputs,\
            encoder_final,\
            embedded = self.encoder(src_var, src_lengths)
        decode_target = trg_var if train else None
        result = self.decoder(
            encoder_final=encoder_final, encoder_outputs=encoder_outputs,
            encoder_mask=input_mask, max_length=trg_max_length,
            trg_var=decode_target, return_attention=return_attention,
            beam_search=not train and trg_var is None)
        if trg_var is not None:
            result['loss'] = self.get_loss(result, trg_var)

        return result

    def get_loss(self, result=None, trg_var=None):
        padding_mask = (trg_var == self.trg_pad_idx)
        if self.use_cuda:
            padding_mask = padding_mask.cuda()
        num_targets = (1 - padding_mask).long().sum().item()
        log_probs = result['log_probs'] # B x S x V
        ce_loss = -(torch.gather(log_probs, 2, trg_var.unsqueeze(2))).squeeze(2)
        if self.fill_padding_mask:
            ce_loss = ce_loss.masked_fill(padding_mask, 0.)
        ce_loss = ce_loss.sum()/num_targets

        return {'loss': ce_loss}
