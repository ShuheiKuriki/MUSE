"""モデル構築"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
from torch import nn
from scipy.stats import truncnorm
from .utils import load_embeddings, normalize_embeddings

class Discriminator(nn.Module):
    """
    n値分類をする分類器
    """

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.emb_dim = params.emb_dim
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout
        self.params = params

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = params.langnum if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.LogSoftmax(dim=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """calculate forward"""
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        output = self.layers(x).view(-1, self.params.langnum)
        return output

class Generator(nn.Module):
    """mapping"""

    def __init__(self, params):
        super(Generator, self).__init__()

        self.emb_dim = params.emb_dim
        self.map_beta = params.map_beta
        self.langnum = params.langnum

        dicos, _embs = [0]*(params.langnum-1), [0]*params.langnum
        for i in range(params.langnum-1):
            dicos[i], _embs[i] = load_embeddings(params, i)
        params.dicos = dicos
        wordnums = [len(dicos[i]) for i in range(params.langnum-1)] + [params.random_vocab]
        self.embs = nn.ModuleList([nn.Embedding(wordnums[i], params.emb_dim, sparse=False) for i in range(self.langnum)])
        for i in range(params.langnum-1):
            self.embs[i].weight.detach().copy_(_embs[i])
        if params.truncated:
            _embs[-1] = torch.from_numpy(truncnorm.rvs(-params.truncated, params.truncated, size=[params.random_vocab, params.emb_dim]))
            self.embs[-1].weight.detach().copy_(_embs[-1])

        self.mappings = nn.ModuleList([nn.Linear(params.emb_dim, params.emb_dim, bias=False) for _ in range(self.langnum-1)])
        if getattr(params, 'map_id_init', True):
            for i in range(self.langnum-1):
                self.mappings[i].weight.detach().copy_(torch.diag(torch.ones(self.emb_dim)))

    def forward(self, x, i):
        """map into target space"""
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        if i == self.langnum-1:
            return x
        return self.mappings[i](x)

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        beta = self.map_beta
        if beta > 0:
            for i in range(self.langnum-1):
                W = self.mappings[i].weight.detach()
                W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

def build_model(params):
    """
    Build all components of the model.
    """
    generator = Generator(params)

    # discriminator
    discriminator = Discriminator(params)
    # cuda
    if params.cuda:
        generator.cuda()
        discriminator.cuda()

    # normalize embeddings
    params.means = [normalize_embeddings(generator.embs[i].weight.detach(), params.normalize_embeddings) for i in range(params.langnum)]

    return generator, discriminator
