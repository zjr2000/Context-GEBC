from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
import pdb


def init_lstm(lstm, bias=True):
    nn.init.orthogonal_(lstm.weight_ih_l0)
    nn.init.orthogonal_(lstm.weight_hh_l0)
    if bias:
        nn.init.ones_(lstm.bias_ih_l0)
        nn.init.ones_(lstm.bias_hh_l0)


def init_bilstm(blstm, bias=True):
    nn.init.orthogonal_(blstm.weight_ih_l0)
    nn.init.orthogonal_(blstm.weight_hh_l0)
    nn.init.orthogonal_(blstm.weight_hh_l0_reverse)
    nn.init.orthogonal_(blstm.weight_ih_l0_reverse)
    if bias:
        nn.init.ones_(blstm.bias_ih_l0)
        nn.init.ones_(blstm.bias_hh_l0)
        nn.init.ones_(blstm.bias_ih_l0_reverse)
        nn.init.ones_(blstm.bias_hh_l0_reverse)

def init_gru(gru, bias=True):
    nn.init.orthogonal_(gru.weight_ih_l0)
    nn.init.orthogonal_(gru.weight_hh_l0)
    if bias:
        nn.init.ones_(gru.bias_ih_l0)
        nn.init.ones_(gru.bias_hh_l0)


class ESGNBasic(nn.Module):
    def __init__(self, opt):
        super(ESGNBasic, self).__init__()
        self.opt = opt
        self.rnn_size = opt.esgn_rnn_size
        self.num_layers = opt.esgn_num_layers
        self.drop_prob = opt.esgn_drop_prob
        self.max_decoding_len = opt.esgn_max_decoding_len
        self.ss_prob = 0.0 # Schedule sampling probability
        self.encoder_rnn = nn.GRU(opt.esgn_input_dim, self.rnn_size, self.num_layers, bias=opt.esgn_enable_rnn_bias, dropout=self.drop_prob, batch_first=True)
        # self.transfer_layer = nn.Linear(self.rnn_size, self.rnn_size)
        # self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob)
        init_gru(self.encoder_rnn, opt.esgn_enable_rnn_bias)

        # initrange = 0.1esgn_position_encoding_size
        # self.embed.weight.data.uniform_(-initrange, initrange)
        # self.logit.bias.data.fill_(0)
        # self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, event):
        rnn_output, h_n = self.encoder_rnn(event)
        h_n = self.dropout(h_n)
        # batch_size, N, rnn_size = h_n.size()
        # h_n = self.transfer_layer(h_n.reshape(-1, rnn_size)).reshape(batch_size, N, rnn_size)
        return h_n

    # def build_loss(self, input, target, mask):
    #     output = input * torch.log(target) + (1-input) * torch.log(1-target)
    #     output = -1 * (output * mask).sum() / (1e-5 + mask.sum())
    #     return output

    def build_loss(self, input, target, mask):
        one_hot = torch.nn.functional.one_hot(target.long(), input.shape[2])
        max_len = input.shape[1]
        output = - (one_hot[:, :max_len] * input * mask[:, :max_len, None]).sum(2).sum(1) / (mask.sum(1) + 1e-6)
        return output.mean()

    def build_rl_loss(self, input, seq, reward):
        input = (input).reshape(-1)
        reward = (reward).reshape(-1)
        mask = (seq > 0).float()
        mask = (torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / (torch.sum(mask) + 1e-6)
        return output

    def forward(self, event, gt_seq):
        # event: shape torch.Size([1, 102, 1024])
        # pos_feats: shape torch.Size([1, 102, 100])
        # gt_seq: torch.Size([1, 5]); tensor([[ 1,  2, 20, 96,  0]], dtype=torch.int32)
        pos_feats = None

        batch_size = event.shape[0]
        hn = self.init_hidden(event[:,2:,:])

        state = (hn, hn)

        outputs = []
        gt_seq = gt_seq.long()
        # event = torch.cat((event, pos_feats), 2)

        for i in range(gt_seq.size(1)):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = event.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = gt_seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = gt_seq[:, i].data.clone()
                    prob_prev = outputs[-1].data # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = gt_seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and gt_seq[:, i].data.sum() == 0:
                break
            xt = event[torch.arange(len(it)),it]
            weight, state = self.core(xt, event, state, isFirstStep=(i==0))
            outputs.append(weight)

        return torch.stack(outputs, dim=1)

    def sample(self, event, opt={}):

        sample_max = opt.get('sample_max', 1)
        temperature = opt.get('temperature', 1.0)
        batch_size = event.shape[0]

        hn = self.init_hidden(event[:,2:,:])
        state = (hn, hn)
        # event = torch.cat((event, pos_feats), 2)

        seq = []
        seqLogprobs = []

        for t in range(self.max_decoding_len + 1):
            if t == 0: # input <bos>
                it = event.data.new(batch_size).long().zero_() + 1
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing
            xt = event[torch.arange(len(it)), it]
            logprobs, state = self.core(xt, event, state, isFirstStep=(t==0))

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished & (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))
        if seq==[] or len(seq)==0:
            return [],[]
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)


    def sample_rerank(self, event, pos_feats, prop_score, opt={}):

        sample_max = opt.get('sample_max', 1)
        batch_size = event.shape[0]
        hn = self.init_hidden(event[:,2:,:])
        state = (hn, hn)
        seq = []
        seqLogprobs = []
        weights = []
        event = torch.cat((event, pos_feats), 2)

        for t in range(self.max_decoding_len + 1):
            w = 0.3
            if t == 0: # input <bos>
                it2 = event.data.new(batch_size).long().zero_() + 1
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs, 1)
                sampleLogprobs2, it2 = torch.max(logprobs2, 1)
                it = it.view(-1).long()
                it2 = it2.view(-1).long()
            else:
                raise AssertionError

            xt2 = event.index_select(dim=1, index=it2).squeeze(1)
            weight, state = self.core(xt2, event, state, isFirstStep=(t==0))
            logprobs = torch.log(weight)
            logprobs2 = logprobs + w * torch.log(prop_score)

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished & (it > 0)
                if unfinished.sum() == 0:
                    break
                it2 = it2 * unfinished.type_as(it)
                seq.append(it2) #seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs2.view(-1))
                weights.append(weight)

        if seq==[] or len(seq)==0:
            return [],[],[]
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1), torch.stack(weights, 1)

class PtrNetCore(nn.Module):

    def __init__(self, opt):
        super(PtrNetCore, self).__init__()
        self.rnn_size = opt.esgn_rnn_size
        self.num_layers = opt.esgn_num_layers
        self.drop_prob = opt.esgn_drop_prob
        self.att_feat_size = opt.esgn_input_dim
        self.att_hid_size = opt.esgn_att_hid_size

        self.opt = opt
        self.rnn = nn.LSTM(opt.esgn_input_dim, self.rnn_size, self.num_layers,
                           bias=opt.esgn_enable_rnn_bias, dropout=self.drop_prob)

        self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        init_lstm(self.rnn, opt.esgn_enable_rnn_bias)

    def forward(self, xt, v, state, isFirstStep=False):
        if not isFirstStep:
            _, state = self.rnn(xt.unsqueeze(0), state)
        att_size = v.numel() // v.size(0) // v.size(-1)
        att = v.view(-1, v.size(-1))

        att = self.ctx2att(att)                             # (batch * att_size) * att_hid_size
        att = att.view(-1, att_size, self.att_hid_size)     # batch * att_size * att_hid_size
        att_h = self.h2att(state[0][-1])                    # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)           # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        # weight = F.sigmoid(dot)
        weight = F.log_softmax(dot, dim=-1)
        return weight, state


class ESGN(ESGNBasic):
    def __init__(self, opt):
        super(ESGN, self).__init__(opt)
        self.core = PtrNetCore(opt)


if __name__ == "__main__":
    from opts import parse_opts
    opt = parse_opts()
    model = ESGN(opt)
