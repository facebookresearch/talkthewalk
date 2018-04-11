
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from torch.autograd import Variable
from dict import START_TOKEN, END_TOKEN, UNK_TOKEN, PAD_TOKEN


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

    def __init__(self, n_lands=10, n_acts=3, hidden_size=128, emb_dim=32,
                 n_layers=1, dropout=0.,
                 bidirectional=False, vocab=None, rnn_type='LSTM', cuda=False):

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
        self.use_cuda = torch.cuda.is_available() and cuda

        self.action_embedding = nn.Embedding(n_acts + n_lands + 4, emb_dim)
        self.observation_embedding = nn.Embedding(n_acts + n_lands + 4, emb_dim)
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
        if self.use_cuda:
            emb_x = emb_x.cuda()
        for i in range(bsz):
            ex = x[i]
            for j in range(len(ex)):
                a_or_o = ex[j]
                if type(a_or_o) is list:  # list of observations
                    emb = torch.Tensor(self.emb_dim).zero_()
                    if self.use_cuda:
                        emb = emb.cuda()
                    for obs in a_or_o:
                        # obs = Variable(torch.LongTensor([obs]))
                        obs = torch.LongTensor([obs])
                        if self.use_cuda:
                            obs = obs.cuda()
                        emb += self.observation_embedding(obs).view(-1)
                else:  # action
                    act = Variable(torch.LongTensor([a_or_o]))
                    if self.use_cuda:
                        act = act.cuda()
                    emb = self.action_embedding(act)
                emb_x[i, j] = emb
        return emb_x


class Decoder(nn.Module):
    """
        Decoder with attention
    """
    def __init__(self, hidden_size=128, emb_dim=128, n_words=0, n_layers=1,
                 dropout=0.1, attn_type='',
                 encoder_is_bidirectional=True, enc_hidden_size=128,
                 pass_hidden_state=False, vocab=None, rnn_type='LSTM',
                 ctx_dim=0, use_prev_word=True, use_dec_state=True,
                 max_length=0, cuda=False):

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
        self.use_cuda = torch.cuda.is_available() and cuda
        self.pass_hidden_state = pass_hidden_state
        self.use_attention = attn_type != ''
        self.attn_type = attn_type
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        if enc_hidden_size == 0:
            enc_hidden_size = hidden_size * 2 if encoder_is_bidirectional else hidden_size
        self.embedding = nn.Embedding(n_words, emb_dim)
        rnn_dim = emb_dim if self.attn_type == '' else emb_dim + enc_hidden_size
        if rnn_type == 'gru':
            self.rnn = nn.GRU(rnn_dim, hidden_size,
                              batch_first=True, dropout=dropout,
                              num_layers=n_layers)
        else:
            self.rnn = nn.LSTM(rnn_dim, hidden_size,
                               batch_first=True, dropout=dropout,
                               num_layers=n_layers)

        if self.use_attention:
            if self.attn_type == 'Bahdanau':
                self.attention = BahdanauAttention(hidden_size, enc_hidden_size)
            else:
                self.attn = nn.Linear(emb_dim + enc_hidden_size, max_length)
                self.attn_combine = nn.Linear(max_length + hidden_size, emb_dim)

        # Projected Final Encoder State
        self.enc_to_dec_h0 = nn.Linear(enc_hidden_size, hidden_size, bias=False)
        self.enc_to_dec_c0 = nn.Linear(enc_hidden_size, hidden_size, bias=False) if rnn_type == 'LSTM' else None
        self.enc_to_dec_h1 = nn.Linear(enc_hidden_size, hidden_size, bias=False) if n_layers > 1 else None
        self.enc_to_dec_c1 = nn.Linear(enc_hidden_size, hidden_size, bias=False) if n_layers > 1 and rnn_type == 'LSTM' else None

        pre_output_input_dim = enc_hidden_size
        if self.use_prev_word:
            pre_output_input_dim += emb_dim
        if self.use_dec_state:
            pre_output_input_dim += hidden_size

        self.pre_output_layer = nn.Linear(pre_output_input_dim, hidden_size)
        # output layer from context vector and current decoder state to n_words
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

        if self.attn_type == 'Bahdanau':
            projected_memory = self.attention.project_memory(encoder_outputs)
            attention_values = encoder_outputs
        # start of sequence = embedding all 0s
        # embedded = Variable(torch.zeros((bsz, 1, self.emb_dim)))
        embedded = Variable(torch.Tensor(bsz, 1, self.emb_dim).fill_(self.vocab.tok2i[START_TOKEN]))
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
                if self.attn_type == 'Bahdanau':
                    query = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]  # [B, D]
                    query = query.unsqueeze(0)  # [1, B, D]
                    alpha = self.attention(query=query,
                                           projected_memory=projected_memory,
                                           mask=encoder_mask)  # (B, 1, T)
                    if return_attention:
                        all_attention_scores.append(alpha)
                    context = alpha.bmm(attention_values)  # (B, 1, D)
                    output = torch.cat((embedded, context), 2) # (B, 1, D+emb_dim)
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
                    output = F.relu(output)
            else:
                output = embedded
            _, hidden = self.rnn(output, hidden)  # hidden (n_layers, B, hsz)

            # predict from (top) RNN hidden state directly
            to_predict = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]  # [B, V]
            context = context if self.attn_type == 'Bahdanau' else output
            if return_states:
                decoder_states.append(to_predict.unsqueeze(0))
            if self.use_prev_word and self.use_dec_state:
                to_predict = torch.cat(
                                (to_predict.unsqueeze(1),
                                    embedded,
                                    context),
                                2)
            elif self.use_prev_word and not self.use_dec_state:
                to_predict = torch.cat((embedded, context), 2)
            elif not self.use_prev_word and not self.use_dec_state:
                to_predict = context
            to_predict = self.pre_output_layer(to_predict)       # (B, 1, D)
            logits = self.output_layer(to_predict)  # (B, 1, V)
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
                 enc_emb_dim=0, dec_emb_dim=0,
                 n_enc_layers=1, n_dec_layers=1, dropout=0., word_dropout=0.,
                 bidirectional=False, attn_type='',
                 pass_hidden_state=True, vocab_src=None, vocab_trg=None,
                 rnn_type=None,
                 ctx_dim=0, use_prev_word=True, use_dec_state=True, max_length=0,
                 cuda=False,
                 **kwargs):

        super(Seq2Seq, self).__init__()
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.hidden_size = hidden_size
        self.enc_emb_dim = enc_emb_dim
        self.dec_emb_dim = dec_emb_dim
        self.use_cuda = torch.cuda.is_available() and cuda

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

        self.encoder = Encoder(n_lands, n_acts, hidden_size, enc_emb_dim,
                               n_enc_layers, dropout, bidirectional, vocab_src,
                               rnn_type, self.use_cuda)
        self.decoder = Decoder(hidden_size, dec_emb_dim, n_words_trg, n_dec_layers,
                               dropout, attn_type, bidirectional,
                               hidden_size, pass_hidden_state, vocab_trg,
                               rnn_type, ctx_dim, use_prev_word,
                               use_dec_state, max_length, self.use_cuda)

    def compute_input_mask(self, src_var):
        mask = torch.LongTensor(len(src_var), len(src_var[0])).fill_(self.src_pad_idx)
        for i in range(len(src_var)):
            for j in range(len(src_var[i])):
                mask[i, j] = 1 if src_var[i][j] != self.src_pad_idx else 0
        return mask

    def forward(self, src_var=None, src_lengths=None, trg_var=None,
                trg_lengths=None, max_length=0, encoder_mask=None,
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

        # input_mask = (src_var != self.src_pad_idx)
        input_mask = encoder_mask
        if input_mask is None:
            input_mask = self.compute_input_mask(src_var)
        if self.use_cuda:
            input_mask = input_mask.cuda()

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
