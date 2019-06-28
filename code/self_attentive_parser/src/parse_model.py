import functools

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.init as init
import pickle

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch_t = torch.cuda
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).pin_memory().cuda(async=True)
else:
    print("Not using CUDA!")
    torch_t = torch
    from torch import from_numpy

device = torch.device("cuda" if use_cuda else "cpu")

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
import chart_helper
import nkutil

import trees

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"

TAG_UNK = "UNK"

# Assumes that these control characters are not present in treebank text
CHAR_UNK = "\0"
CHAR_START_SENTENCE = "\1"
CHAR_START_WORD = "\2"
CHAR_STOP_WORD = "\3"
CHAR_STOP_SENTENCE = "\4"

BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    }

# Path to elmo data files
elmo_path = "/homes/ttmt001/transitory/self-attentive-parser/data"
bert_path = "/homes/ttmt001/transitory/self-attentive-parser/data"
fisher_path = "/g/ssli/projects/disfluencies/ttmt001/fisher_disf"
glove_pretrained_path = "/homes/ttmt001/transitory/GloVe-1.2"

class BatchIndices:
    """
    Batch indices container class (used to implement packed batches)
    """
    def __init__(self, batch_idxs_np):
        self.batch_idxs_np = batch_idxs_np
        # Note that the torch copy will be on GPU if use_cuda is set
        self.batch_idxs_torch = from_numpy(batch_idxs_np)
        self.batch_size = int(1 + np.max(batch_idxs_np))

        batch_idxs_np_extra = np.concatenate([[-1], batch_idxs_np, [-1]])
        self.boundaries_np = np.nonzero(batch_idxs_np_extra[1:] != \
                batch_idxs_np_extra[:-1])[0]
        self.seq_lens_np = self.boundaries_np[1:] - self.boundaries_np[:-1]
        assert len(self.seq_lens_np) == self.batch_size
        self.max_len = int(np.max(self.boundaries_np[1:] - \
                self.boundaries_np[:-1]))


class FeatureDropoutFunction(nn.functional._functions.dropout.InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, batch_idxs, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))

        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = input.new().resize_(batch_idxs.batch_size, \
                    input.size(1))
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            ctx.noise = ctx.noise[batch_idxs.batch_idxs_torch, :]
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            # Got rid of Variable here
            return grad_output.mul(ctx.noise), None, None, None, None
        else:
            return grad_output, None, None, None, None


class FeatureDropout(nn.Module):
    """
    Feature-level dropout: takes an input of size len x num_features and drops
    each feature with probabibility p. A feature is dropped across the full
    portion of the input that corresponds to a single batch element.
    """
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input, batch_idxs):
        return FeatureDropoutFunction.apply(input, batch_idxs, self.p, self.training, self.inplace)


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3, affine=True):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
            self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(-1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        if self.affine:
            ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        # NOTE(nikita): the t2t code does the following instead, with eps=1e-6
        # However, I currently have no reason to believe that this difference in
        # implementation matters.
        # mu = torch.mean(z, keepdim=True, dim=-1)
        # variance = torch.mean((z - mu.expand_as(z))**2, keepdim=True, dim=-1)
        # ln_out = (z - mu.expand_as(z)) * torch.rsqrt(variance + self.eps).expand_as(z)
        # ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = d_model ** 0.5
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        # q: [batch, slot, feat]
        # k: [batch, slot, feat]
        # v: [batch, slot, feat]

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            # Got rid of "data" here
            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        # Note that this makes the distribution not sum to 1. At some point it
        # may be worth researching whether this is the right way to apply
        # dropout to the attention.
        # Note that the t2t code also applies dropout in this manner
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module
    """

    def __init__(self, n_head, d_model, d_k, d_v, 
            residual_dropout=0.1, 
            attention_dropout=0.1, 
            d_positional=None,
            d_speech=0):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_speech = d_speech

        if d_positional is None:
            self.partitioned = False
        else:
            self.partitioned = True

        if self.partitioned:
            self.d_content = d_model - d_positional - d_speech
            self.d_positional = d_positional

            if d_speech > 0:
                self.w_qs1 = nn.Parameter(torch_t.FloatTensor(n_head, \
                        self.d_content, d_k - 2 * (d_k // 3) ))
                self.w_ks1 = nn.Parameter(torch_t.FloatTensor(n_head, \
                        self.d_content, d_k - 2 * (d_k // 3) ))
                self.w_vs1 = nn.Parameter(torch_t.FloatTensor(n_head, \
                        self.d_content, d_v - 2 * (d_v // 3) ))

                self.w_qs2 = nn.Parameter(torch_t.FloatTensor(n_head, \
                        self.d_positional, d_k // 3))
                self.w_ks2 = nn.Parameter(torch_t.FloatTensor(n_head, \
                        self.d_positional, d_k // 3))
                self.w_vs2 = nn.Parameter(torch_t.FloatTensor(n_head, \
                        self.d_positional, d_v // 3))

                self.w_qs3 = nn.Parameter(torch_t.FloatTensor(n_head, \
                        self.d_speech, d_k // 3))
                self.w_ks3 = nn.Parameter(torch_t.FloatTensor(n_head, \
                        self.d_speech, d_k // 3))
                self.w_vs3 = nn.Parameter(torch_t.FloatTensor(n_head, \
                        self.d_speech, d_v // 3))

                init.xavier_normal_(self.w_qs1)
                init.xavier_normal_(self.w_ks1)
                init.xavier_normal_(self.w_vs1)

                init.xavier_normal_(self.w_qs2)
                init.xavier_normal_(self.w_ks2)
                init.xavier_normal_(self.w_vs2)

                init.xavier_normal_(self.w_qs3)
                init.xavier_normal_(self.w_ks3)
                init.xavier_normal_(self.w_vs3)

            else:
                self.w_qs1 = nn.Parameter(torch_t.FloatTensor(n_head, \
                        self.d_content, d_k // 2))
                self.w_ks1 = nn.Parameter(torch_t.FloatTensor(n_head, \
                        self.d_content, d_k // 2))
                self.w_vs1 = nn.Parameter(torch_t.FloatTensor(n_head, \
                        self.d_content, d_v // 2))

                self.w_qs2 = nn.Parameter(torch_t.FloatTensor(n_head, \
                        self.d_positional, d_k // 2))
                self.w_ks2 = nn.Parameter(torch_t.FloatTensor(n_head, \
                        self.d_positional, d_k // 2))
                self.w_vs2 = nn.Parameter(torch_t.FloatTensor(n_head, \
                        self.d_positional, d_v // 2))

                init.xavier_normal_(self.w_qs1)
                init.xavier_normal_(self.w_ks1)
                init.xavier_normal_(self.w_vs1)

                init.xavier_normal_(self.w_qs2)
                init.xavier_normal_(self.w_ks2)
                init.xavier_normal_(self.w_vs2)

        else:
            self.w_qs = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_k))
            self.w_ks = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_k))
            self.w_vs = nn.Parameter(torch_t.FloatTensor(n_head, d_model, d_v))

            init.xavier_normal_(self.w_qs)
            init.xavier_normal_(self.w_ks)
            init.xavier_normal_(self.w_vs)

        self.attention = ScaledDotProductAttention(d_model, \
                attention_dropout=attention_dropout)
        self.layer_norm = LayerNormalization(d_model)

        if not self.partitioned:
            # The lack of a bias term here is consistent with the t2t code; 
            # in my experiments I have never observed this making a difference.
            self.proj = nn.Linear(n_head*d_v, d_model, bias=False)
        else:
            if self.d_speech > 0:
                self.proj1 = nn.Linear(n_head * (d_v - 2 * (d_v // 3)), \
                        self.d_content, bias=False)
                self.proj2 = nn.Linear(n_head * (d_v // 3), self.d_positional, \
                    bias=False)
                self.proj3 = nn.Linear(n_head * (d_v // 3), self.d_speech, \
                    bias=False)
            else:
                self.proj1 = nn.Linear(n_head*(d_v//2), self.d_content, \
                    bias=False)
                self.proj2 = nn.Linear(n_head*(d_v//2), self.d_positional, \
                    bias=False)

        self.residual_dropout = FeatureDropout(residual_dropout)

    def split_qkv_packed(self, inp, qk_inp=None):
        v_inp_repeated = inp.repeat(self.n_head, 1).view(self.n_head, \
                -1, inp.size(-1)) # n_head x len_inp x d_model
        if qk_inp is None:
            qk_inp_repeated = v_inp_repeated
        else:
            qk_inp_repeated = qk_inp.repeat(self.n_head, 1).view(self.n_head, \
                    -1, qk_inp.size(-1))

        if not self.partitioned:
            q_s = torch.bmm(qk_inp_repeated, self.w_qs) # n_head x len_inp x d_k
            k_s = torch.bmm(qk_inp_repeated, self.w_ks) # n_head x len_inp x d_k
            v_s = torch.bmm(v_inp_repeated, self.w_vs) # n_head x len_inp x d_v
        else:
            if self.d_speech > 0:
                q_s = torch.cat([
                    torch.bmm(qk_inp_repeated[:,:,:self.d_content], self.w_qs1),
                    torch.bmm(qk_inp_repeated[:,:,self.d_content:self.d_content+self.d_positional], self.w_qs2),
                    torch.bmm(qk_inp_repeated[:,:,self.d_content+self.d_positional:], self.w_qs3),
                    ], -1)
                k_s = torch.cat([
                    torch.bmm(qk_inp_repeated[:,:,:self.d_content], self.w_ks1),
                    torch.bmm(qk_inp_repeated[:,:,self.d_content:self.d_content+self.d_positional], self.w_ks2),
                    torch.bmm(qk_inp_repeated[:,:,self.d_content+self.d_positional:], self.w_ks3),
                    ], -1)
                v_s = torch.cat([
                    torch.bmm(v_inp_repeated[:,:,:self.d_content], self.w_vs1),
                    torch.bmm(v_inp_repeated[:,:,self.d_content:self.d_content+self.d_positional], self.w_vs2),
                    torch.bmm(v_inp_repeated[:,:,self.d_content+self.d_positional:], self.w_vs3),
                    ], -1)
            else:
                q_s = torch.cat([
                    torch.bmm(qk_inp_repeated[:,:,:self.d_content], self.w_qs1),
                    torch.bmm(qk_inp_repeated[:,:,self.d_content:], self.w_qs2),
                    ], -1)
                k_s = torch.cat([
                    torch.bmm(qk_inp_repeated[:,:,:self.d_content], self.w_ks1),
                    torch.bmm(qk_inp_repeated[:,:,self.d_content:], self.w_ks2),
                    ], -1)
                v_s = torch.cat([
                    torch.bmm(v_inp_repeated[:,:,:self.d_content], self.w_vs1),
                    torch.bmm(v_inp_repeated[:,:,self.d_content:], self.w_vs2),
                    ], -1)
        return q_s, k_s, v_s

    def pad_and_rearrange(self, q_s, k_s, v_s, batch_idxs):
        # Input is padded representation: n_head x len_inp x d
        # Output is packed representation: (n_head * mb_size) x len_padded x d
        # (along with masks for the attention and output)
        n_head = self.n_head
        d_k, d_v = self.d_k, self.d_v

        len_padded = batch_idxs.max_len
        mb_size = batch_idxs.batch_size
        q_padded = q_s.data.new(n_head, mb_size, len_padded, d_k).fill_(0.)
        k_padded = k_s.data.new(n_head, mb_size, len_padded, d_k).fill_(0.)
        v_padded = v_s.data.new(n_head, mb_size, len_padded, d_v).fill_(0.)
        # Don't need to wrap in Variable() in torch 0.4.x
        #q_padded = Variable(q_padded)
        #k_padded = Variable(k_padded)
        #v_padded = Variable(v_padded)
        #invalid_mask = torch_t.ByteTensor(mb_size, len_padded).fill_(True)
        invalid_mask = q_s.new_ones((mb_size, len_padded), dtype=torch.uint8)

        for i, (start, end) in enumerate(zip(batch_idxs.boundaries_np[:-1], \
                batch_idxs.boundaries_np[1:])):
            q_padded[:,i,:end-start,:] = q_s[:,start:end,:]
            k_padded[:,i,:end-start,:] = k_s[:,start:end,:]
            v_padded[:,i,:end-start,:] = v_s[:,start:end,:]
            invalid_mask[i, :end-start].fill_(False)

        attn_mask = invalid_mask.unsqueeze(1).expand(mb_size, \
                len_padded, len_padded).repeat(n_head, 1, 1)
        #output_mask = (~invalid_mask).repeat(n_head, 1).unsqueeze(-1)
        # populate to same dim as outputs_padded, for torch-0.4.x
        output_mask = (~invalid_mask).repeat(n_head, 1)

        return(
            q_padded.view(-1, len_padded, d_k),
            k_padded.view(-1, len_padded, d_k),
            v_padded.view(-1, len_padded, d_v),
            attn_mask,
            output_mask
            )

    def combine_v(self, outputs):
        # Combine attention information from the different heads
        n_head = self.n_head
        outputs = outputs.view(n_head, -1, self.d_v)

        if not self.partitioned:
            # Switch from n_head x len_inp x d_v to len_inp x (n_head * d_v)
            outputs = torch.transpose(outputs, 0, 1).contiguous().view(-1, \
                    n_head * self.d_v)

            # Project back to residual size
            outputs = self.proj(outputs)
        else:
            if self.d_speech > 0:
                d_v1 = self.d_v - 2 * (self.d_v // 3)
                outputs1 = outputs[:,:,:d_v1]
                outputs2 = outputs[:,:,d_v1:d_v1 + (self.d_v//3)]
                outputs3 = outputs[:,:,d_v1 + (self.d_v//3):]
                outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(\
                        -1, n_head * d_v1)
                outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(\
                        -1, n_head * (self.d_v // 3))
                outputs3 = torch.transpose(outputs3, 0, 1).contiguous().view(\
                        -1, n_head * (self.d_v // 3))
                outputs = torch.cat([
                    self.proj1(outputs1),
                    self.proj2(outputs2),
                    self.proj3(outputs3),
                    ], -1)

            else:
                d_v1 = self.d_v // 2
                outputs1 = outputs[:,:,:d_v1]
                outputs2 = outputs[:,:,d_v1:]
                outputs1 = torch.transpose(outputs1, 0, 1).contiguous().view(
                        -1, n_head * d_v1)
                outputs2 = torch.transpose(outputs2, 0, 1).contiguous().view(
                        -1, n_head * d_v1)
                outputs = torch.cat([
                    self.proj1(outputs1),
                    self.proj2(outputs2),
                    ], -1)

        return outputs

    def forward(self, inp, batch_idxs, qk_inp=None):
        residual = inp

        # While still using a packed representation, project to obtain the
        # query/key/value for each head
        q_s, k_s, v_s = self.split_qkv_packed(inp, qk_inp=qk_inp)

        # Switch to padded representation, perform attention, then switch back
        q_padded, k_padded, v_padded, attn_mask, output_mask = \
                self.pad_and_rearrange(q_s, k_s, v_s, batch_idxs)

        outputs_padded, attns_padded = self.attention(
            q_padded, k_padded, v_padded,
            attn_mask=attn_mask,
            )
        outputs = outputs_padded[output_mask]
        outputs = self.combine_v(outputs)

        outputs = self.residual_dropout(outputs, batch_idxs)

        return self.layer_norm(outputs + residual), attns_padded


class PositionwiseFeedForward(nn.Module):
    """
    A position-wise feed forward module.

    Projects to a higher-dimensional space before applying ReLU, then projects
    back.
    """

    def __init__(self, d_hid, d_ff, relu_dropout=0.1, residual_dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_hid, d_ff)
        self.w_2 = nn.Linear(d_ff, d_hid)

        self.layer_norm = LayerNormalization(d_hid)
        # The t2t code on github uses relu dropout, even though the transformer
        # paper describes residual dropout only. We implement relu dropout
        # because we always have the option to set it to zero.
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()


    def forward(self, x, batch_idxs):
        residual = x

        output = self.w_1(x)
        output = self.relu_dropout(self.relu(output), batch_idxs)
        output = self.w_2(output)

        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)


class PartitionedFeedForward(nn.Module):
    def __init__(self, d_hid, d_ff, d_positional, d_speech, 
            relu_dropout=0.1, residual_dropout=0.1):
        super().__init__()
        self.d_content = d_hid - d_positional - d_speech
        self.d_speech = d_speech
        self.d_positional = d_positional
        self.w_1c = nn.Linear(self.d_content, d_ff//2)
        self.w_2c = nn.Linear(d_ff//2, self.d_content)
        self.w_1p = nn.Linear(d_positional, d_ff//2)
        self.w_2p = nn.Linear(d_ff//2, d_positional)
        if d_speech > 0:
            self.w_1s = nn.Linear(d_speech, d_ff//2)
            self.w_2s = nn.Linear(d_ff//2, d_speech)
        
        self.layer_norm = LayerNormalization(d_hid)
        # The t2t code on github uses relu dropout, even though the transformer
        # paper describes residual dropout only. We implement relu dropout
        # because we always have the option to set it to zero.
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()

    def forward(self, x, batch_idxs):
        residual = x
        xc = x[:, :self.d_content]
        xp = x[:, self.d_content:self.d_content + self.d_positional]
        outputc = self.w_1c(xc)
        outputc = self.relu_dropout(self.relu(outputc), batch_idxs)
        outputc = self.w_2c(outputc)

        outputp = self.w_1p(xp)
        outputp = self.relu_dropout(self.relu(outputp), batch_idxs)
        outputp = self.w_2p(outputp)

        if self.d_speech > 0:
            xs = x[:, self.d_content + self.d_positional:]
            outputs = self.w_1s(xs)
            outputs = self.relu_dropout(self.relu(outputs), batch_idxs)
            outputs = self.w_2s(outputs)
            output = torch.cat([outputc, outputp, outputs], -1)
        else:
            output = torch.cat([outputc, outputp], -1)

        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)


class PartitionedPositionwiseFeedForward(nn.Module):
    def __init__(self, d_hid, d_ff, d_positional, 
            relu_dropout=0.1, residual_dropout=0.1):
        super().__init__()
        self.d_content = d_hid - d_positional
        self.w_1c = nn.Linear(self.d_content, d_ff//2)
        self.w_1p = nn.Linear(d_positional, d_ff//2)
        self.w_2c = nn.Linear(d_ff//2, self.d_content)
        self.w_2p = nn.Linear(d_ff//2, d_positional)
        self.layer_norm = LayerNormalization(d_hid)
        # The t2t code on github uses relu dropout, even though the transformer
        # paper describes residual dropout only. We implement relu dropout
        # because we always have the option to set it to zero.
        self.relu_dropout = FeatureDropout(relu_dropout)
        self.residual_dropout = FeatureDropout(residual_dropout)
        self.relu = nn.ReLU()

    def forward(self, x, batch_idxs):
        residual = x
        xc = x[:, :self.d_content]
        xp = x[:, self.d_content:]

        outputc = self.w_1c(xc)
        outputc = self.relu_dropout(self.relu(outputc), batch_idxs)
        outputc = self.w_2c(outputc)

        outputp = self.w_1p(xp)
        outputp = self.relu_dropout(self.relu(outputp), batch_idxs)
        outputp = self.w_2p(outputp)

        output = torch.cat([outputc, outputp], -1)

        output = self.residual_dropout(output, batch_idxs)
        return self.layer_norm(output + residual)


class MultiLevelEmbedding(nn.Module):
    def __init__(self,
            num_embeddings_list,
            d_embedding,
            d_positional=None,
            d_speech=0,
            max_len=300,
            normalize=True,
            dropout=0.1,
            timing_dropout=0.0,
            emb_dropouts_list=None,
            extra_content_dropout=None,
            **kwargs):
        super().__init__()

        self.d_embedding = d_embedding
        self.partitioned = d_positional is not None

        if self.partitioned:
            self.d_positional = d_positional
            self.d_speech = d_speech
            self.d_content = self.d_embedding-self.d_positional-self.d_speech
        else:
            self.d_positional = self.d_embedding
            self.d_content = self.d_embedding

        #print("Dimensions partition MLE: ", d_speech, d_positional)

        if emb_dropouts_list is None:
            emb_dropouts_list = [0.0] * len(num_embeddings_list)
        assert len(emb_dropouts_list) == len(num_embeddings_list)

        embs = []
        emb_dropouts = []
        for i, (num_embeddings, emb_dropout) in \
                enumerate(zip(num_embeddings_list, emb_dropouts_list)):
            emb = nn.Embedding(num_embeddings, self.d_content, **kwargs)
            embs.append(emb)
            emb_dropout = FeatureDropout(emb_dropout)
            emb_dropouts.append(emb_dropout)
        self.embs = nn.ModuleList(embs)
        self.emb_dropouts = nn.ModuleList(emb_dropouts)

        if extra_content_dropout is not None:
            self.extra_content_dropout = FeatureDropout(extra_content_dropout)
        else:
            self.extra_content_dropout = None

        if normalize:
            self.layer_norm = LayerNormalization(d_embedding)
        else:
            self.layer_norm = lambda x: x

        self.dropout = FeatureDropout(dropout)
        self.timing_dropout = FeatureDropout(timing_dropout)

        # Learned embeddings
        self.position_table = nn.Parameter(torch_t.FloatTensor(max_len, \
                self.d_positional))
        init.normal_(self.position_table)


    def forward(self, xs, batch_idxs, extra_content_annotations=None,
            speech_content_annotations=None):
        content_annotations = [
            emb_dropout(emb(x), batch_idxs)
            for x, emb, emb_dropout in zip(xs, self.embs, self.emb_dropouts)
            ]
        
        content_annotations = sum(content_annotations)

        if extra_content_annotations is not None:
            if self.extra_content_dropout is not None:
                content_annotations += self.extra_content_dropout(\
                        extra_content_annotations, batch_idxs)
            else:
                content_annotations += extra_content_annotations

        timing_signal = torch.cat([self.position_table[:seq_len,:] for \
                seq_len in batch_idxs.seq_lens_np], dim=0)
        timing_signal = self.timing_dropout(timing_signal, batch_idxs)

        if speech_content_annotations is not None:
            if self.extra_content_dropout is not None:
                speech_content_annotations = self.extra_content_dropout(\
                        speech_content_annotations, batch_idxs)

            # Combine the content and timing signals
            if self.partitioned:
                annotations = torch.cat([content_annotations, timing_signal, 
                    speech_content_annotations], 1)
            else:
                annotations = content_annotations + timing_signal + \
                        speech_content_annotations

        else:
            # Combine the content and timing signals
            if self.partitioned:
                annotations = torch.cat([content_annotations, timing_signal], 1)
            else:
                annotations = content_annotations + timing_signal 

        # TODO(nikita): reconsider the use of layernorm here
        annotations = self.layer_norm(self.dropout(annotations, batch_idxs))

        return annotations, timing_signal, batch_idxs


class CharacterLSTM(nn.Module):
    def __init__(self, num_embeddings, d_embedding, d_out,
            char_dropout=0.0,
            normalize=False,
            **kwargs):
        super().__init__()

        self.d_embedding = d_embedding
        self.d_out = d_out

        self.lstm = nn.LSTM(self.d_embedding, self.d_out // 2, \
                num_layers=1, bidirectional=True)

        self.emb = nn.Embedding(num_embeddings, self.d_embedding, **kwargs)
        #TODO(nikita): feature-level dropout?
        self.char_dropout = nn.Dropout(char_dropout)

        if normalize:
            print("This experiment: layer-normalizing after character LSTM")
            self.layer_norm = LayerNormalization(self.d_out, affine=False)
        else:
            self.layer_norm = lambda x: x

    def forward(self, chars_padded_np, word_lens_np, batch_idxs):
        # copy to ensure nonnegative stride for successful transfer to pytorch
        decreasing_idxs_np = np.argsort(word_lens_np)[::-1].copy()
        # Got rid of Variable in the next 2 lines
        decreasing_idxs_torch = from_numpy(decreasing_idxs_np)
        chars_padded = from_numpy(chars_padded_np[decreasing_idxs_np])

        word_lens = from_numpy(word_lens_np[decreasing_idxs_np])

        inp_sorted = nn.utils.rnn.pack_padded_sequence(chars_padded, \
                word_lens_np[decreasing_idxs_np], batch_first=True)
        inp_sorted_emb = nn.utils.rnn.PackedSequence(
            self.char_dropout(self.emb(inp_sorted.data)),
            inp_sorted.batch_sizes)
        _, (lstm_out, _) = self.lstm(inp_sorted_emb)

        lstm_out = torch.cat([lstm_out[0], lstm_out[1]], -1)

        # Undo sorting by decreasing word length
        res = torch.zeros_like(lstm_out)
        res.index_copy_(0, decreasing_idxs_torch, lstm_out)

        res = self.layer_norm(res)
        return res

def get_elmo_class():
    # Avoid a hard dependency by only importing Elmo if it's being used
    from allennlp.modules.elmo import Elmo
    return Elmo

def get_elmo_class_old():
    # Avoid a hard dependency by only importing Elmo if it's being used
    from allennlp.modules.elmo import Elmo

    class ModElmo(Elmo):
       def forward(self, inputs):
            """
            Unlike Elmo.forward, return vector representations for 
            bos/eos tokens

            This modified version does not support extra tensor dimensions

            Parameters
            ----------
            inputs : ``torch.autograd.Variable``
                Shape ``(batch_size, timesteps, 50)`` of character 
                ids representing the current batch.

            Returns
            -------
            Dict with keys:
            ``'elmo_representations'``: ``List[torch.autograd.Variable]``
                A ``num_output_representations`` list of ELMo representations 
                for the input sequence.
                Each representation is 
                shape ``(batch_size, timesteps + 2, embedding_dim)``
            ``'mask'``:  ``torch.autograd.Variable``
                Shape ``(batch_size, timesteps + 2)`` 
                long tensor with sequence mask.
            """
            # reshape the input if needed
            original_shape = inputs.size()
            timesteps, num_characters = original_shape[-2:]
            assert len(original_shape) == 3, "Only 3D tensors supported here"
            reshaped_inputs = inputs

            # run the biLM
            bilm_output = self._elmo_lstm(reshaped_inputs)
            layer_activations = bilm_output['activations']
            mask_with_bos_eos = bilm_output['mask']

            # compute the elmo representations
            representations = []
            for i in range(len(self._scalar_mixes)):
                scalar_mix = getattr(self, 'scalar_mix_{}'.format(i))
                representation_with_bos_eos = \
                        scalar_mix(layer_activations, mask_with_bos_eos)
                # We don't remove bos/eos here!
                representations.append(self._dropout(\
                        representation_with_bos_eos))

            mask = mask_with_bos_eos
            elmo_representations = representations

            return {'elmo_representations': elmo_representations, 'mask': mask}
    return ModElmo

def get_bert(bert_model, bert_do_lower_case, freeze=False):
    # Avoid a hard dependency on BERT by only importing it if it's being used
    from pytorch_pretrained_bert import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained(bert_model, \
            do_lower_case=bert_do_lower_case)
    bert_model_path = os.path.join(bert_path, bert_model + ".tar.gz")
    
    bert = BertModel.from_pretrained(bert_model_path)
    if freeze:
        for param in bert.parameters():
            param.requires_grad = False

    return tokenizer, bert

def get_glove(model_type='fisher', freeze=False):
    # vocab includes START, STOP, and UNK
    # 300 is dimension for all glove vectors
    if model_type == 'pretrained':
        #vocab_path = os.path.join(glove_pretrained_path, "glove.6B.300d.vocab")
        #vectors_path = os.path.join(glove_pretrained_path, \
        #        "glove.6B.300d.vectors")
        #f_vocab = open(vocab_path, 'r')
        #word2idx = pickle.load(f_vocab)
        #f_vocab.close()
        #f_vec = open(vectors_path, 'r')
        #vectors = pickle.load(f_vec)
        #f_vec.close()
        model_path = os.path.join(glove_pretrained_path, "glove.6B.300d.txt")
    else:
        model_path = os.path.join(fisher_path, "fisher-vectors.txt")

    idx = 0
    word2idx = {}
    f = open(model_path).readlines()
    num_vocab = len(f)
    vectors = np.zeros((num_vocab,300))
    for l in f:
        line = l.strip().split()
        word = line[0]
        vec = np.array(line[1:]).astype(np.float)
        vectors[idx, :] = vec
        word2idx[word] = idx
        idx += 1

    word2idx[START] = idx 
    word2idx[STOP] = idx + 1
    if "<unk>" not in word2idx:
        word2idx["<unk>"] = idx + 2
        num_extra = 3
    else:
        num_extra = 2
    vectors = np.vstack([vectors, np.random.normal(size=(num_extra, 300))])

    glove = nn.Embedding(len(word2idx), 300)
    glove.load_state_dict({'weight': torch.Tensor(vectors)})

    if freeze:
        for param in glove.parameters():
            param.requires_grad = False

    return word2idx, glove


class Encoder(nn.Module):
    def __init__(self, embedding,
                    num_layers=1, num_heads=2, d_kv = 32, d_ff=1024,
                    d_positional=None, d_speech=0,
                    num_layers_position_only=0,
                    relu_dropout=0.1, 
                    residual_dropout=0.1, 
                    attention_dropout=0.1):
        super().__init__()
        # Don't assume ownership of the embedding as a submodule.
        # TODO(nikita): what's the right thing to do here?
        self.embedding_container = [embedding]
        d_model = embedding.d_embedding

        d_k = d_v = d_kv

        self.stacks = []
        for i in range(num_layers):
            attn = MultiHeadAttention(num_heads, d_model, d_k, d_v, \
                    residual_dropout=residual_dropout, \
                    attention_dropout=attention_dropout, \
                    d_positional=d_positional, d_speech=d_speech)
            if d_positional is None:
                ff = PositionwiseFeedForward(d_model, d_ff, \
                        relu_dropout=relu_dropout, \
                        residual_dropout=residual_dropout)
            else:
                ff = PartitionedFeedForward(d_model, d_ff, \
                        d_positional, d_speech, \
                        relu_dropout=relu_dropout, \
                        residual_dropout=residual_dropout)

            self.add_module("attn_{}".format(i), attn)
            self.add_module("ff_{}".format(i), ff)
            self.stacks.append((attn, ff))

        self.num_layers_position_only = num_layers_position_only
        if self.num_layers_position_only > 0:
            assert d_positional is None, \
                ("num_layers_position_only and partitioned are incompatible")

    def forward(self, xs, batch_idxs, extra_content_annotations=None, 
            speech_content_annotations=None):
        emb = self.embedding_container[0]
        res, timing_signal, batch_idxs = emb(xs, batch_idxs, \
                extra_content_annotations=extra_content_annotations,
                speech_content_annotations=speech_content_annotations)

        for i, (attn, ff) in enumerate(self.stacks):
            if i >= self.num_layers_position_only:
                res, current_attns = attn(res, batch_idxs)
            else:
                res, current_attns = attn(res, batch_idxs, qk_inp=timing_signal)
            res = ff(res, batch_idxs)

        return res, batch_idxs


# Extenstion of NKChartParser; mainly in the speech encoding module
class SpeechFeatureEncoder(nn.Module):
    def __init__(self,
            feature_sizes,
            d_out,
            conv_sizes=[5, 10, 25, 50],
            num_conv=32,
            d_pause_embedding=4,  
            speech_dropout=0.0):
        super().__init__()

        self.d_pause_embedding = d_pause_embedding
        self.d_out = d_out
        self.speech_dropout = nn.Dropout(speech_dropout)
        self.feature_sizes = feature_sizes
        self.d_in = 0
        self.num_conv = num_conv
        self.conv_sizes = conv_sizes

        if 'pause' in feature_sizes.keys():
            self.emb = nn.Embedding(self.feature_sizes['pause'], \
                    self.d_pause_embedding)
            self.d_in += self.d_pause_embedding

        if 'frames' in feature_sizes.keys():
            conv_modules = []
            feat_dim = feature_sizes['frames']
            word_length = feature_sizes['word_length']
            for filter_size in conv_sizes:
                kernel_size = (filter_size, feat_dim)
                pool_kernel = (word_length - filter_size + 1, 1)
                filter_conv = nn.Sequential(
                        nn.Conv2d(1, num_conv, kernel_size),
                        nn.ReLU(),
                        nn.MaxPool2d(pool_kernel, 1)
                        )
                conv_modules.append(filter_conv.to(device))

            self.conv_modules = nn.ModuleList(conv_modules)

            self.d_conv = self.num_conv * len(self.conv_sizes)
            self.d_in += self.d_conv

        if 'scalars' in feature_sizes.keys():
            self.d_scalars = self.feature_sizes['scalars']
            self.d_in += self.d_scalars

        self.speech_projection = nn.Linear(self.d_in, self.d_out, bias=True)

    def forward(self, processed_features):
        pause_features, frame_features, scalar_features = processed_features
        all_features = []
        if len(pause_features) > 0:
            all_features.append(self.emb(pause_features))
        if len(scalar_features) > 0:
            all_features.append(scalar_features.transpose(0, 1))
        if len(frame_features) > 0:
            conv_outputs = [convolve(frame_features) for \
                    convolve in self.conv_modules]
            conv_outputs = [x.squeeze(-1).squeeze(-1) for x in conv_outputs]
            conv_outputs = torch.cat(conv_outputs, -1)
            assert conv_outputs.shape[1] == self.d_conv
            all_features.append(conv_outputs)
        
        all_features = torch.cat(all_features, -1)
        assert all_features.shape[1] == self.d_in
        res = self.speech_dropout(self.speech_projection(all_features))
        return res


# Probably won't need the char-level encoding since this is speech
# But keeping it here for baseline and possible extra feature purposes
class SpeechParser(nn.Module):
    # We never actually call forward() end-to-end as is typical for pytorch
    # modules, but this inheritance brings in good stuff like state dict
    # management.
    def __init__(
            self,
            tag_vocab,
            word_vocab,
            label_vocab,
            char_vocab,
            pause_vocab,
            speech_features,
            hparams,
    ):
        super().__init__()
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("__class__")
        self.spec['hparams'] = hparams.to_dict()

        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab
        self.pause_vocab = pause_vocab

        self.d_model = hparams.d_model
        self.partitioned = hparams.partitioned

        if self.partitioned:
            if speech_features is not None:
                self.d_positional = hparams.d_model // 3
                self.d_speech = hparams.d_model // 3
            else:
                self.d_positional = hparams.d_model // 2
                self.d_speech = 0
        else:
            self.d_positional = None
            if speech_features is not None:
                self.d_speech = hparams.d_model 
            else: 
                self.d_speech = 0

        self.d_content = (self.d_model - self.d_positional - self.d_speech) \
                if self.partitioned else self.d_model

        num_embeddings_map = {
            'tags': tag_vocab.size,
            'words': word_vocab.size,
            'chars': char_vocab.size,
            'pause': pause_vocab.size,
        }
        emb_dropouts_map = {
            'tags': hparams.tag_emb_dropout,
            'words': hparams.word_emb_dropout,
        }

        self.emb_types = []
        if hparams.use_tags:
            self.emb_types.append('tags')
        if hparams.use_words:
            self.emb_types.append('words')

        self.use_tags = hparams.use_tags

        self.morpho_emb_dropout = None
        if hparams.use_chars_lstm or hparams.use_chars_concat or \
                hparams.use_elmo or hparams.use_bert or hparams.use_bert_only \
                or hparams.use_glove_pretrained or hparams.use_glove_fisher:
            self.morpho_emb_dropout = hparams.morpho_emb_dropout
        else:
            assert self.emb_types, ("Need at least one of: use_tags, \
                    use_words, use_chars_lstm, use_chars_concat, \
                    use_elmo, use_bert, use_glove_[fisher|pretrained]")

        self.char_encoder = None
        self.char_embedding = None
        self.elmo = None
        self.bert = None
        self.glove = None
        self.speech_encoder = None
        self.fixed_word_length = hparams.fixed_word_length

        if speech_features is not None:
            feature_sizes = {}
            if 'pause' in speech_features:
                feature_sizes['pause'] = self.pause_vocab.size
            if 'duration' in speech_features or 'f0coefs' in speech_features:
                feature_sizes['scalars'] = \
                      hparams.d_duration * int('duration' in speech_features) \
                    + hparams.d_f0coefs * int('f0coefs' in speech_features)
            if 'fbank' in speech_features or 'mfcc' in speech_features or \
                    'pitch' in speech_features:
                feature_sizes['word_length'] = self.fixed_word_length
                feature_sizes['frames'] = \
                    hparams.d_mfcc * int('mfcc' in speech_features) \
                    + hparams.d_fbank * int('fbank' in speech_features) \
                    + hparams.d_pitch * int('pitch' in speech_features)

            self.speech_encoder = SpeechFeatureEncoder(feature_sizes, 
                    self.d_speech, d_pause_embedding=hparams.d_pause_emb)

        if hparams.use_chars_lstm:
            assert not hparams.use_chars_concat, ("use_chars_lstm and \
                    use_chars_concat are mutually exclusive")
            assert not hparams.use_elmo, ("use_chars_lstm and \
                    use_elmo are mutually exclusive")
            self.char_encoder = CharacterLSTM(
                num_embeddings_map['chars'],
                hparams.d_char_emb,
                self.d_content,
                char_dropout=hparams.char_lstm_input_dropout,
            )
        elif hparams.use_chars_concat:
            assert not hparams.use_elmo, ("use_chars_concat and use_elmo \
                    are mutually exclusive")
            self.num_chars_flat = self.d_content // hparams.d_char_emb
            assert self.num_chars_flat >= 2, ("incompatible settings of \
                    d_model/partitioned and d_char_emb")
            assert self.num_chars_flat == (self.d_content/hparams.d_char_emb),\
                    ("d_char_emb does not evenly divide model size")

            self.char_embedding = nn.Embedding(
                num_embeddings_map['chars'],
                hparams.d_char_emb,
                )

        elif hparams.use_glove_pretrained:
            assert not hparams.use_glove_fisher, ("use_glove_pretrained and \
                    use_glove_fisher are mutually exclusive")
            assert not hparams.use_elmo, ("use_glove_pretrained and use_elmo \
                    are mutually exclusive")
            assert not hparams.use_elmo, ("use_glove_pretrained and use_elmo \
                    are mutually exclusive")
            assert not hparams.use_bert, ("use_glove_pretrained and use_bert \
                    are mutually exclusive")
            assert not hparams.use_bert_only, ("use_glove_pretrained and \
                    use_bert_only are mutually exclusive")
            d_glove = 300
            self.glove_vocab, self.glove = get_glove('pretrained', \
                    hparams.freeze)
            self.project_glove = nn.Linear(d_glove, self.d_content, bias=False)

        elif hparams.use_glove_fisher:
            assert not hparams.use_glove_pretrained, ("use_glove_fisher and \
                    use_glove_pretrained are mutually exclusive")
            assert not hparams.use_elmo, ("use_glove_fisher and use_elmo \
                    are mutually exclusive")
            assert not hparams.use_elmo, ("use_glove_fisher and use_elmo \
                    are mutually exclusive")
            assert not hparams.use_bert, ("use_glove_fisher and use_bert \
                    are mutually exclusive")
            assert not hparams.use_bert_only, ("use_glove_fisher and \
                    use_bert_only are mutually exclusive")
            d_glove = 300
            self.glove_vocab, self.glove = get_glove('fisher', hparams.freeze)
            self.project_glove = nn.Linear(d_glove, self.d_content, bias=False)

        elif hparams.use_elmo:
            assert not hparams.use_bert, ("use_elmo and use_bert are \
                    mutually exclusive")
            assert not hparams.use_bert_only, ("use_elmo and use_bert_only are \
                    mutually exclusive")
            self.elmo = get_elmo_class()(
                options_file = os.path.join(elmo_path, \
                        "elmo_2x4096_512_2048cnn_2xhighway_options.json"),
                weight_file = os.path.join(elmo_path, \
                        "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"),
                num_output_representations=1,
                requires_grad=not hparams.freeze,
                do_layer_norm=False,
                keep_sentence_boundaries=True,
                dropout=hparams.elmo_dropout,
                )
            d_elmo_annotations = 1024

            # Don't train gamma parameter for ELMo - the projection can do any
            # necessary scaling
            self.elmo.scalar_mix_0.gamma.requires_grad = False

            # Reshapes the embeddings to match the model dimension, and making
            # the projection trainable appears to improve parsing accuracy
            self.project_elmo = nn.Linear(d_elmo_annotations, 
                    self.d_content, 
                    bias=False)

        elif hparams.use_bert or hparams.use_bert_only:
            self.bert_tokenizer, self.bert = get_bert(hparams.bert_model, \
                    hparams.bert_do_lower_case, hparams.freeze)
            # NOTE: original code had this, I'm not dealing with other languages
            self.bert_transliterate = None

            d_bert_annotations = self.bert.pooler.dense.in_features
            self.bert_max_len = self.bert.embeddings.position_embeddings.num_embeddings

            if hparams.use_bert_only:
                self.project_bert = nn.Linear(d_bert_annotations, \
                        hparams.d_model, bias=False)
            else:
                self.project_bert = nn.Linear(d_bert_annotations, \
                        self.d_content, bias=False)

        if hparams.use_bert_only:
            self.embedding = None
            self.encoder = None

        else:
            self.embedding = MultiLevelEmbedding(
                [num_embeddings_map[emb_type] for emb_type in self.emb_types],
                hparams.d_model,
                d_positional=self.d_positional,
                d_speech=self.d_speech,
                dropout=hparams.embedding_dropout,
                timing_dropout=hparams.timing_dropout,
                emb_dropouts_list=[emb_dropouts_map[emb_type] for emb_type \
                        in self.emb_types],
                extra_content_dropout=self.morpho_emb_dropout,
                max_len=hparams.sentence_max_len,
            )

            self.encoder = Encoder(
                self.embedding,
                num_layers=hparams.num_layers,
                num_heads=hparams.num_heads,
                d_kv=hparams.d_kv,
                d_ff=hparams.d_ff,
                d_positional=self.d_positional,
                d_speech=self.d_speech,
                num_layers_position_only=hparams.num_layers_position_only,
                relu_dropout=hparams.relu_dropout,
                residual_dropout=hparams.residual_dropout,
                attention_dropout=hparams.attention_dropout,
            )

        self.f_label = nn.Sequential(
            nn.Linear(hparams.d_model, hparams.d_label_hidden),
            LayerNormalization(hparams.d_label_hidden),
            nn.ReLU(),
            nn.Linear(hparams.d_label_hidden, label_vocab.size - 1),
            )

        if hparams.predict_tags:
            assert not hparams.use_tags, ("use_tags and predict_tags are \
                    mutually exclusive")
            self.f_tag = nn.Sequential(
                nn.Linear(hparams.d_model, hparams.d_tag_hidden),
                LayerNormalization(hparams.d_tag_hidden),
                nn.ReLU(),
                nn.Linear(hparams.d_tag_hidden, tag_vocab.size),
                )
            self.tag_loss_scale = hparams.tag_loss_scale
        else:
            self.f_tag = None

        # Put model on correponding device (cuda or cpu)
        self.to(device)

    @property
    def model(self):
        return self.state_dict()

    @classmethod
    def from_spec(cls, spec, model):
        spec = spec.copy()
        hparams = spec['hparams']
        if 'sentence_max_len' not in hparams:
            hparams['sentence_max_len'] = 300
        if 'use_elmo' not in hparams:
            hparams['use_elmo'] = False
        if 'elmo_dropout' not in hparams:
            hparams['elmo_dropout'] = 0.5
        if 'use_bert' not in hparams:
            hparams['use_bert'] = False
        if 'use_glove_fisher' not in hparams:
            hparams['use_glove_fisher'] = False
        if 'use_glove_pretrained' not in hparams:
            hparams['use_glove_pretrained'] = False
        if 'use_bert_only' not in hparams:
            hparams['use_bert_only'] = False
        if 'freeze' not in hparams:
            hparams['freeze'] = False
        if 'predict_tags' not in hparams:
            hparams['predict_tags'] = False
        if 'bert_transliterate' not in hparams:
            hparams['bert_transliterate'] = ""

        spec['hparams'] = nkutil.HParams(**hparams)
        res = cls(**spec)
        if use_cuda:
            res.cpu()
        if not hparams['use_elmo']:
            res.load_state_dict(model)
        else:
            state = {k: v for k,v in res.state_dict().items() if k not in model}
            state.update(model)
            res.load_state_dict(state)
        if use_cuda:
            res.cuda()
        return res
    
    def process_sent_frames(self, sent_partition, sent_frames):
        feat_dim = sent_frames.shape[0]
        speech_frames = []
        for frame_idx in sent_partition:
            center_frame = int((frame_idx[0] + frame_idx[1])/2)
            start_idx = center_frame - int(self.fixed_word_length/2)
            end_idx = center_frame + int(self.fixed_word_length/2)
            raw_word_frames = sent_frames[:, frame_idx[0]:frame_idx[1]]
            # feat_dim * number of frames
            raw_count = raw_word_frames.shape[1]
            if raw_count > self.fixed_word_length:
                # too many frames, choose wisely
                this_word_frames = sent_frames[:, frame_idx[0]:frame_idx[1]]
                extra_ratio = int(raw_count/self.fixed_word_length)
                if extra_ratio < 2:  # delete things in the middle
                    mask = np.ones(raw_count, dtype=bool)
                    num_extra = raw_count - self.fixed_word_length
                    not_include = range(center_frame-num_extra,
                                        center_frame+num_extra)[::2]
                    # need to offset by beginning frame
                    not_include = [x-frame_idx[0] for x in not_include]
                    mask[not_include] = False
                else:  # too big, just sample
                    mask = np.zeros(raw_count, dtype=bool)
                    include = range(frame_idx[0], frame_idx[1])[::extra_ratio]
                    include = [x-frame_idx[0] for x in include]
                    if len(include) > self.fixed_word_length:
                        # still too many frames
                        num_current = len(include)
                        sub_extra = num_current - self.fixed_word_length
                        num_start = int((num_current - sub_extra)/2)
                        not_include = include[num_start:num_start+sub_extra]
                        for ni in not_include:
                            include.remove(ni)
                    mask[include] = True
                this_word_frames = this_word_frames[:, mask]
            else:  # not enough frames, choose frames extending from center
                this_word_frames = sent_frames[:, max(0, start_idx):end_idx]
                if this_word_frames.shape[1] == 0:
                    # make 0 if no frame info
                    this_word_frames = np.zeros((feat_dim,
                                                 self.fixed_word_length))
                if start_idx < 0 and \
                        this_word_frames.shape[1] < self.fixed_word_length:
                    this_word_frames = np.hstack(
                        [np.zeros((feat_dim, -start_idx)), this_word_frames])

                # still not enough frames
                if this_word_frames.shape[1] < self.fixed_word_length:
                    num_more = self.fixed_word_length-this_word_frames.shape[1]
                    this_word_frames = np.hstack(
                        [this_word_frames, np.zeros((feat_dim, num_more))])
            # flip frames within word
            speech_frames.append(this_word_frames)
        
        # Add dummy word features for START and STOP
        sent_frame_features = [np.zeros((feat_dim, self.fixed_word_length))] \
            + speech_frames + [np.zeros((feat_dim, self.fixed_word_length))] 
        return sent_frame_features

    def prep_features(self, sent_ids, sfeatures):
        pause_features = []
        frame_features = []
        scalar_features = []
        for sent in sent_ids:
            sent_features = sfeatures[sent]
            if 'pause' in sent_features.keys():
                sent_pauses = [START] + [str(i) for i in \
                        sent_features['pause']] + [STOP]
                sent_pauses = [self.pause_vocab.index(x) for x in sent_pauses]
                pause_features += sent_pauses
            if 'scalars' in sent_features.keys():
                sent_scalars = sent_features['scalars']
                feat_dim = sent_scalars.shape[0]
                sent_scalar_feat = np.hstack([np.zeros((feat_dim, 1)), \
                        sent_scalars, \
                        np.zeros((feat_dim, 1))])
                scalar_features.append(sent_scalar_feat)
            if 'frames' in sent_features.keys():
                assert 'partition' in sent_features.keys(), \
                        ("Must provide partition as a feature")
                sent_partition = sent_features['partition']
                sent_frames = sent_features['frames']
                sent_frame_features = self.process_sent_frames(sent_partition, \
                        sent_frames)
                # sent_frame_features: list of [feat_dim, fixed_word_length]
                sent_frame_features = [torch.Tensor(word_frames.T).unsqueeze(0)\
                        for word_frames in sent_frame_features]
                frame_features += sent_frame_features
        
        if pause_features:
            pause_features = torch.LongTensor(pause_features).to(device)
        
        if frame_features:
            # need frame feats of shape: [batch, 1, fixed_word_length, feat_dim]
            # second dimension is num input channel, defaults to 1        
            frame_features = torch.cat(frame_features, 0)
            frame_features = frame_features.unsqueeze(1).to(device)

        if scalar_features:
            scalar_features = np.hstack(scalar_features)
            scalar_features = torch.Tensor(scalar_features).to(device)
        
        return pause_features, frame_features, scalar_features

    def split_batch(self, sentences, golds, sent_ids, subbatch_max_tokens=3000):
        lens = [len(sentence) + 2 for sentence in sentences]
        lens = np.asarray(lens, dtype=int)
        lens_argsort = np.argsort(lens).tolist()
        num_subbatches = 0
        subbatch_size = 1
        while lens_argsort:
            if (subbatch_size==len(lens_argsort)) or (subbatch_size * \
                    lens[lens_argsort[subbatch_size]]>subbatch_max_tokens):
                yield [sentences[i] for i in lens_argsort[:subbatch_size]], \
                        [golds[i] for i in lens_argsort[:subbatch_size]], \
                        [sent_ids[i] for i in lens_argsort[:subbatch_size]]
                lens_argsort = lens_argsort[subbatch_size:]
                num_subbatches += 1
                subbatch_size = 1
            else:
                subbatch_size += 1

    def parse(self, sentence, gold=None):
        tree_list, loss_list = self.parse_batch([sentence], [gold] \
                if gold is not None else None)
        return tree_list[0], loss_list[0]

    def parse_batch(self, sentences, sent_ids, sfeatures, golds=None, \
            return_label_scores_charts=False, backoff=False):
        is_train = golds is not None
        self.train(is_train)
        torch.set_grad_enabled(is_train)

        if golds is None:
            golds = [None] * len(sentences)

        packed_len = sum([(len(sentence) + 2) for sentence in sentences])

        i = 0
        tag_idxs = np.zeros(packed_len, dtype=int)
        word_idxs = np.zeros(packed_len, dtype=int)
        batch_idxs = np.zeros(packed_len, dtype=int)
        for snum, sentence in enumerate(sentences):
            for (tag, word) in [(START, START)] + sentence + [(STOP, STOP)]:
                tag_idxs[i] = 0 if (not self.use_tags and self.f_tag is None) \
                        else self.tag_vocab.index_or_unk(tag, TAG_UNK)
                if word not in (START, STOP):
                    count = self.word_vocab.count(word)
                    if not count or \
                            (is_train and np.random.rand() < 1 / (1 + count)):
                        word = UNK
                word_idxs[i] = self.word_vocab.index(word)
                batch_idxs[i] = snum
                i += 1
        assert i == packed_len

        batch_idxs = BatchIndices(batch_idxs)

        emb_idxs_map = {
            'tags': tag_idxs,
            'words': word_idxs,
        }
        # Got rid of Variable wrapping here
        emb_idxs = [torch.LongTensor(emb_idxs_map[emb_type]).to(device) \
                for emb_type in self.emb_types]
        
        if is_train and self.f_tag is not None:
            gold_tag_idxs = from_numpy(emb_idxs_map['tags'])

        extra_content_annotations = None
        speech_content_annotations = None
        
        if self.speech_encoder is not None:
            # process speeech feature: pad, extract frames etcs
            processed_features = self.prep_features(sent_ids, sfeatures)
            speech_content_annotations = self.speech_encoder(processed_features)

        if self.char_encoder is not None:
            assert isinstance(self.char_encoder, CharacterLSTM)
            max_word_len = max([max([len(word) for tag, word in sentence]) \
                    for sentence in sentences])
            # Add 2 for start/stop tokens
            max_word_len = max(max_word_len, 3) + 2
            char_idxs_encoder = np.zeros((packed_len, max_word_len), dtype=int)
            word_lens_encoder = np.zeros(packed_len, dtype=int)

            i = 0
            for snum, sentence in enumerate(sentences):
                for wordnum, (tag, word) in \
                    enumerate([(START, START)] + sentence + [(STOP, STOP)]):
                    j = 0
                    char_idxs_encoder[i, j] = self.char_vocab.index(
                            CHAR_START_WORD)
                    j += 1
                    if word in (START, STOP):
                        char_idxs_encoder[i, j:j+3] = self.char_vocab.index(
                            CHAR_START_SENTENCE if (word == START) \
                                    else CHAR_STOP_SENTENCE)
                        j += 3
                    else:
                        for char in word:
                            char_idxs_encoder[i, j] = \
                                    self.char_vocab.index_or_unk(char, CHAR_UNK)
                            j += 1
                    char_idxs_encoder[i, j] = \
                            self.char_vocab.index(CHAR_STOP_WORD)
                    word_lens_encoder[i] = j + 1
                    i += 1
            assert i == packed_len

            extra_content_annotations = self.char_encoder(char_idxs_encoder, \
                    word_lens_encoder, batch_idxs)

        elif self.char_embedding is not None:
            char_idxs_encoder = np.zeros((packed_len, self.num_chars_flat), \
                    dtype=int)

            i = 0
            for snum, sentence in enumerate(sentences):
                for wordnum, (tag, word) in enumerate([(START, START)] \
                        + sentence + [(STOP, STOP)]):
                    if word == START:
                        char_idxs_encoder[i, :] = self.char_vocab.index(
                                CHAR_START_SENTENCE)
                    elif word == STOP:
                        char_idxs_encoder[i, :] = self.char_vocab.index(
                                CHAR_STOP_SENTENCE)
                    else:
                        word_chars = \
                            (([self.char_vocab.index(CHAR_START_WORD)] \
                                * self.num_chars_flat) \
                                + [self.char_vocab.index_or_unk(char, CHAR_UNK)\
                                for char in word] \
                                + ([self.char_vocab.index(CHAR_STOP_WORD)] \
                                * self.num_chars_flat))
                        char_idxs_encoder[i, :self.num_chars_flat//2] = \
                            word_chars[self.num_chars_flat:self.num_chars_flat \
                            + self.num_chars_flat//2]
                        char_idxs_encoder[i, self.num_chars_flat//2:] = \
                            word_chars[::-1][self.num_chars_flat:self.num_chars_flat + self.num_chars_flat//2]
                    i += 1
            assert i == packed_len

            # Got rid of Variable here
            char_idxs_encoder = from_numpy(char_idxs_encoder)

            extra_content_annotations = self.char_embedding(char_idxs_encoder)
            extra_content_annotations = extra_content_annotations.view(-1, \
                    self.num_chars_flat * self.char_embedding.embedding_dim)

        if self.elmo is not None:
            # See https://github.com/allenai/allennlp/blob/c3c3549887a6b1fb0bc8abf77bc820a3ab97f788/allennlp/data/token_indexers/elmo_indexer.py#L61
            # ELMO_START_SENTENCE = 256
            # ELMO_STOP_SENTENCE = 257
            ELMO_START_WORD = 258
            ELMO_STOP_WORD = 259
            ELMO_CHAR_PAD = 260

            # Sentence start/stop tokens are added inside the ELMo module
            max_sentence_len = max([(len(sentence)) for sentence in sentences])
            max_word_len = 50
            char_idxs_encoder = np.zeros((len(sentences), \
                    max_sentence_len, max_word_len), dtype=int)

            for snum, sentence in enumerate(sentences):
                for wordnum, (tag, word) in enumerate(sentence):
                    char_idxs_encoder[snum, wordnum, :] = ELMO_CHAR_PAD

                    j = 0
                    char_idxs_encoder[snum, wordnum, j] = ELMO_START_WORD
                    j += 1
                    assert word not in (START, STOP)
                    for char_id in word.encode('utf-8', 'ignore')[:(max_word_len-2)]:
                        char_idxs_encoder[snum, wordnum, j] = char_id
                        j += 1
                    char_idxs_encoder[snum, wordnum, j] = ELMO_STOP_WORD

                    # +1 for masking 
                    # (everything that stays 0 is past the end of the sentence)
                    char_idxs_encoder[snum, wordnum, :] += 1

            # Got rid of Variable here
            char_idxs_encoder = from_numpy(char_idxs_encoder)

            elmo_out = self.elmo.forward(char_idxs_encoder)
            elmo_rep0 = elmo_out['elmo_representations'][0]
            elmo_mask = elmo_out['mask']

            d_elmo = elmo_rep0.shape[-1]

            # pytorch 0.4.x requires same dimension for mask
            elmo_annotations_packed = elmo_rep0[
                    elmo_mask.byte()].view(packed_len, -1)

            # Apply projection to match dimensionality
            extra_content_annotations = self.project_elmo(
                    elmo_annotations_packed)

        elif self.glove is not None:
            packed_len = sum([(len(sentence) + 2) for sentence in sentences])

            i = 0
            word_idxs = np.zeros(packed_len, dtype=int)
            for snum, sentence in enumerate(sentences):
                for _, word in [(START, START)] + sentence + [(STOP, STOP)]:
                    if word not in self.glove_vocab:
                        word = "<unk>"
                    word_idxs[i] = self.glove_vocab[word]
                    i += 1
            assert i == packed_len

            emb_idxs = torch.LongTensor(word_idxs).to(device) 
            glove_annotations_packed = self.glove(emb_idxs) 
            extra_content_annotations = self.project_glove(
                    glove_annotations_packed)

        elif self.bert is not None:
            all_input_ids = np.zeros((len(sentences), self.bert_max_len), \
                    dtype=int)
            all_input_mask = np.zeros((len(sentences), self.bert_max_len), \
                    dtype=int)
            all_word_start_mask = np.zeros((len(sentences), self.bert_max_len),\
                    dtype=int)
            all_word_end_mask = np.zeros((len(sentences), self.bert_max_len), \
                    dtype=int)

            subword_max_len = 0
            for snum, sentence in enumerate(sentences):
                tokens = []
                word_start_mask = []
                word_end_mask = []

                tokens.append("[CLS]")
                word_start_mask.append(1)
                word_end_mask.append(1)

                if self.bert_transliterate is None:
                    cleaned_words = []
                    for _, word in sentence:
                        word = BERT_TOKEN_MAPPING.get(word, word)
                        if word == "n't" and cleaned_words:
                            cleaned_words[-1] = cleaned_words[-1] + "n"
                            word = "'t"
                        cleaned_words.append(word)
                else:
                    # When transliterating, assume that the token mapping is
                    # taken care of elsewhere
                    cleaned_words = [self.bert_transliterate(word) \
                            for _, word in sentence]

                for word in cleaned_words:
                    temp_word_tokens = self.bert_tokenizer.tokenize(word)
                    word_tokens = []
                    for w in temp_word_tokens:
                        if w not in self.bert_tokenizer.vocab:
                            w = "[UNK]"
                        word_tokens.append(w)
                    for _ in range(len(word_tokens)):
                        word_start_mask.append(0)
                        word_end_mask.append(0)
                    word_start_mask[len(tokens)] = 1
                    word_end_mask[-1] = 1
                    tokens.extend(word_tokens)
                tokens.append("[SEP]")
                word_start_mask.append(1)
                word_end_mask.append(1)

                input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. 
                # Only real tokens are attended to.
                input_mask = [1] * len(input_ids)

                subword_max_len = max(subword_max_len, len(input_ids))

                all_input_ids[snum, :len(input_ids)] = input_ids
                all_input_mask[snum, :len(input_mask)] = input_mask
                all_word_start_mask[snum,:len(word_start_mask)]=word_start_mask
                all_word_end_mask[snum, :len(word_end_mask)] = word_end_mask

            all_input_ids = from_numpy(np.ascontiguousarray(\
                    all_input_ids[:, :subword_max_len]))
            all_input_mask = from_numpy(np.ascontiguousarray(\
                    all_input_mask[:, :subword_max_len]))
            all_word_start_mask = from_numpy(np.ascontiguousarray(\
                    all_word_start_mask[:, :subword_max_len]))
            all_word_end_mask = from_numpy(np.ascontiguousarray(\
                    all_word_end_mask[:, :subword_max_len]))
            all_encoder_layers, _ = self.bert(all_input_ids, \
                    attention_mask=all_input_mask)
            del _
            features = all_encoder_layers[-1]

            if self.encoder is not None:
                features_packed = features.masked_select(\
                    all_word_end_mask.to(torch.uint8).unsqueeze(-1)).reshape(\
                    -1, features.shape[-1])

                # For now, just project the features from the last word piece 
                # in each word
                extra_content_annotations = self.project_bert(features_packed)

        ########################################
        # End of extra_content_annotation cases
        ########################################
        if self.encoder is not None:
            annotations, _ = self.encoder(emb_idxs, batch_idxs, 
                extra_content_annotations=extra_content_annotations, 
                speech_content_annotations=speech_content_annotations)

            if self.partitioned:
                # Rearrange the annotations to ensure that the transition to
                # fenceposts captures an even split between position and content.
                if speech_content_annotations is not None:
                    annotations = torch.cat([
                        annotations[:, 0::3],
                        annotations[:, 1::3],
                        annotations[:, 2::3],
                    ], 1)
                else:
                    annotations = torch.cat([
                        annotations[:, 0::2],
                        annotations[:, 1::2],
                    ], 1)
            
            if self.f_tag is not None:
                tag_annotations = annotations

            # TT: This part just throws away the left half of bottom row 
            # and the right half of top row; and basically makes
            # left half "forward" vectors and right half "backward" vectors
            fencepost_annotations = torch.cat([
                annotations[:-1, :self.d_model//2],
                annotations[1:, self.d_model//2:],
                ], 1)
            fencepost_annotations_start = fencepost_annotations
            fencepost_annotations_end = fencepost_annotations
        else:
            assert self.bert is not None
            features = self.project_bert(features)
            fencepost_annotations_start = features.masked_select(all_word_start_mask.to(torch.uint8).unsqueeze(-1)).reshape(-1, features.shape[-1])
            fencepost_annotations_end = features.masked_select(all_word_end_mask.to(torch.uint8).unsqueeze(-1)).reshape(-1, features.shape[-1])
            if self.f_tag is not None:
                tag_annotations = fencepost_annotations_end

        if self.f_tag is not None:
            tag_logits = self.f_tag(tag_annotations)
            if is_train:
                tag_loss = self.tag_loss_scale*nn.functional.cross_entropy(\
                        tag_logits, gold_tag_idxs, reduction='sum')

        # Note that the subtraction above creates fenceposts at sentence
        # boundaries, which are not used by our parser. Hence subtract 1
        # when creating fp_endpoints
        fp_startpoints = batch_idxs.boundaries_np[:-1]
        fp_endpoints = batch_idxs.boundaries_np[1:] - 1

        # Just return the charts, for ensembling
        if return_label_scores_charts:
            charts = []
            for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
                chart = self.label_scores_from_annotations(\
                        fencepost_annotations_start[start:end,:], \
                        fencepost_annotations_end[start:end,:])
                charts.append(chart.cpu().data.numpy())
            return charts

        if not is_train:
            trees = []
            scores = []
            if self.f_tag is not None:
                # Note that tag_logits includes tag predictions 
                # for start/stop tokens
                tag_idxs = torch.argmax(tag_logits, -1).cpu()
                per_sentence_tag_idxs = torch.split_with_sizes(tag_idxs, \
                        [len(sentence) + 2 for sentence in sentences])
                per_sentence_tags = [[self.tag_vocab.value(idx) for idx \
                        in idxs[1:-1]] for idxs in per_sentence_tag_idxs]

            for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
                sentence = sentences[i]
                if self.f_tag is not None:
                    sentence = list(zip(per_sentence_tags[i], \
                            [x[1] for x in sentence]))
                tree, score = self.parse_from_annotations(\
                        fencepost_annotations_start[start:end,:], \
                        fencepost_annotations_end[start:end,:], \
                        sentence, golds[i])
                trees.append(tree)
                scores.append(score)
            return trees, scores

        # During training time, the forward pass needs to be computed for every
        # cell of the chart, but the backward pass only needs to be computed for
        # cells in either the predicted or the gold parse tree. It's slightly
        # faster to duplicate the forward pass for a subset of the chart than it
        # is to perform a backward pass that doesn't take advantage of sparsity.
        # Since this code is not undergoing algorithmic changes, it makes sense
        # to include the optimization even though it may only be a 10% speedup.
        # Note that no dropout occurs in the label portion of the network
        pis = []
        pjs = []
        plabels = []
        paugment_total = 0.0
        num_p = 0
        gis = []
        gjs = []
        glabels = []
        with torch.no_grad():
            for i, (start, end) in enumerate(zip(fp_startpoints, fp_endpoints)):
                p_i, p_j, p_label, p_augment, g_i, g_j, g_label = \
                        self.parse_from_annotations(\
                        fencepost_annotations_start[start:end,:], \
                        fencepost_annotations_end[start:end,:], \
                        sentences[i], golds[i])
                paugment_total += p_augment
                num_p += p_i.shape[0]
                pis.append(p_i + start)
                pjs.append(p_j + start)
                gis.append(g_i + start)
                gjs.append(g_j + start)
                plabels.append(p_label)
                glabels.append(g_label)

        cells_i = from_numpy(np.concatenate(pis + gis))
        cells_j = from_numpy(np.concatenate(pjs + gjs))
        cells_label = from_numpy(np.concatenate(plabels + glabels))

        cells_label_scores = self.f_label(fencepost_annotations_end[cells_j] \
                - fencepost_annotations_start[cells_i])
        cells_label_scores = torch.cat([
                cells_label_scores.new_zeros((cells_label_scores.size(0), 1)),
                cells_label_scores
                ], 1)
        cells_scores = torch.gather(cells_label_scores, 1, cells_label[:, None])
        loss = cells_scores[:num_p].sum() - cells_scores[num_p:].sum() \
                + paugment_total

        if self.f_tag is not None:
            return None, (loss, tag_loss)
        else:
            return None, loss


    def label_scores_from_annotations(self, fencepost_annotations_start, \
            fencepost_annotations_end):
        # Note that the bias added to the final layer norm is useless because
        # this subtraction gets rid of it
        span_features = (torch.unsqueeze(fencepost_annotations_end, 0)
                         - torch.unsqueeze(fencepost_annotations_start, 1))

        label_scores_chart = self.f_label(span_features)
        label_scores_chart = torch.cat([
            label_scores_chart.new_zeros((label_scores_chart.size(0), \
            label_scores_chart.size(1), 1)), label_scores_chart \
            ], 2)
        return label_scores_chart

    def parse_from_annotations(self, fencepost_annotations_start, \
            fencepost_annotations_end, sentence, gold=None, backoff=False):
        is_train = gold is not None
        label_scores_chart = self.label_scores_from_annotations(\
                fencepost_annotations_start, fencepost_annotations_end)
        label_scores_chart_np = label_scores_chart.cpu().data.numpy()

        if is_train:
            decoder_args = dict(
                sentence_len=len(sentence),
                label_scores_chart=label_scores_chart_np,
                gold=gold,
                label_vocab=self.label_vocab,
                is_train=is_train,
                backoff=False)

            p_score, p_i, p_j, p_label, p_augment = chart_helper.decode(False, \
                    **decoder_args)
            g_score, g_i, g_j, g_label, g_augment = chart_helper.decode(True, \
                    **decoder_args)
            return p_i, p_j, p_label, p_augment, g_i, g_j, g_label
        else:
            return self.decode_from_chart(sentence, label_scores_chart_np)

    def decode_from_chart_batch(self, sentences, charts_np, golds=None, backoff=False):
        trees = []
        scores = []
        if golds is None:
            golds = [None] * len(sentences)
        for sentence, chart_np, gold in zip(sentences, charts_np, golds):
            tree, score = self.decode_from_chart(sentence, chart_np, gold)
            trees.append(tree)
            scores.append(score)
        return trees, scores

    def decode_from_chart(self, sentence, chart_np, gold=None, backoff=False):
        decoder_args = dict(
            sentence_len=len(sentence),
            label_scores_chart=chart_np,
            gold=gold,
            label_vocab=self.label_vocab,
            is_train=False,
            backoff=backoff)

        force_gold = (gold is not None)

        # The optimized cython decoder implementation doesn't actually
        # generate trees, only scores and span indices. When converting to a
        # tree, we assume that the indices follow a preorder traversal.
        score, p_i, p_j, p_label, _ = chart_helper.decode(\
                force_gold, **decoder_args)
        last_splits = []
        idx = -1
        def make_tree():
            nonlocal idx
            idx += 1
            i, j, label_idx = p_i[idx], p_j[idx], p_label[idx]
            label = self.label_vocab.value(label_idx)
            if (i + 1) >= j:
                tag, word = sentence[i]
                tree = trees.LeafParseNode(int(i), tag, word)
                if label:
                    tree = trees.InternalParseNode(label, [tree])
                return [tree]
            else:
                left_trees = make_tree()
                right_trees = make_tree()
                children = left_trees + right_trees
                if label:
                    return [trees.InternalParseNode(label, children)]
                else:
                    return children

        tree = make_tree()[0]
        return tree, score


