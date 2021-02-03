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
from torch.nn import functional as F
from scipy.stats import truncnorm
from .utils import load_embeddings, normalize_embeddings

class Discriminator(nn.Module):
    """
    Containing n dicscriminators. Each network discriminates their lang respectively
    """
    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.emb_dim = params.emb_dim
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        models = []
        for l in range(params.langnum):
            layers = [nn.Dropout(self.dis_input_dropout)]
            for i in range(self.dis_layers + 1):
                input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
                output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
                layers.append(nn.Linear(input_dim, output_dim))
                if i < self.dis_layers:
                    layers.append(nn.LeakyReLU(0.2))
                    layers.append(nn.Dropout(self.dis_dropout))
            layers.append(nn.Sigmoid())
            models.append(nn.Sequential(*layers))
        self.models = nn.ModuleList(models)

    def forward(self, x, i):
        """
        make prediction by discriminator for lang i
        """
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.models[i](x).view(-1)

class Mapping(nn.Module):
    """
    mapping for source langs
    """

    def __init__(self, params):
        super(Mapping, self).__init__()

        self.map_beta = params.map_beta
        self.langnum = params.langnum

        self.models = nn.ModuleList([nn.Linear(params.emb_dim, params.emb_dim, bias=False) for _ in range(self.langnum-1)])
        if getattr(params, 'map_id_init', True):
            for i in range(self.langnum-1):
                self.models[i].weight.data = torch.diag(torch.ones(params.emb_dim))

    def forward(self, x, i, rev=False):
        """
        map into target space
        """
        if i % self.langnum == self.langnum-1: return x
        if not rev: return self.models[i](x)
        return F.linear(x, self.models[i].weight.t())

    def orthogonalize(self):
        """
        Orthogonalize the all mappings
        """
        beta = self.map_beta
        for i in range(self.langnum-1):
            W = self.models[i].weight.detach()
            self.models[i].weight.data = (1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W))

class Embedding(nn.Module):
    """
    embeddings
    """

    def __init__(self, params):
        super(Embedding, self).__init__()

        self.emb_dim = params.emb_dim
        self.langnum = params.langnum

        # set src embeddings and dicos
        params.dicos, _embs = [0]*self.langnum, [0]*self.langnum
        for i in range(self.langnum-1):
            params.dicos[i], _embs[i] = load_embeddings(params, i)

        self.embs = nn.ModuleList([nn.Embedding(len(params.dicos[i]), params.emb_dim, sparse=False) for i in range(self.langnum-1)])

        for i in range(self.langnum-1): self.embs[i].weight.data = _embs[i]

        # set tgt embedding and dico
        if params.random_vocab:
            params.dicos[-1], _embs[-1] = [0]*params.random_vocab, self.initialize_random(params)
        else:
            params.dicos[-1], _embs[-1] = load_embeddings(params, params.langnum-1)

        self.embs.append(nn.Embedding(len(params.dicos[-1]), params.emb_dim, sparse=False))
        self.embs[-1].weight.data = _embs[-1]

        if params.learnable: self.embs[-1].weight.requires_grad = True

    def initialize_random(self, params):
        """
        initialize random vectors
        """
        emb = torch.randn(params.random_vocab, params.emb_dim) / (params.emb_dim**.5)
        if params.emb_init == 'norm_mean':
            mean_norms = [torch.norm(self.embs[l].weight.data, dim=1, keepdim=True).expand_as(self.embs[l].weight.data) for l in range(self.langnum-1)]
            norm_mean = torch.mean(torch.cat(mean_norms).view(self.langnum, -1, params.emb_dim), dim=0)
            emb *= norm_mean[:params.random_vocab] / torch.mean(norm_mean[:params.random_vocab]) * params.emb_norm
        elif params.emb_init == 'load':
            emb = torch.load(params.emb_file).to('cpu')
        elif params.emb_init == 'uniform':
            emb *= params.emb_norm
        return emb

def build_model(params, with_dis=True):
    """
    Build all components of the model
    """
    mapping = Mapping(params).to(params.device)
    embedding = Embedding(params).to(params.device)

    # discriminator
    discriminators = Discriminator(params).to(params.device) if with_dis else None

    # normalize embeddings
    params.means = [normalize_embeddings(embedding.embs[i].weight.detach(), params.normalize_embeddings) for i in range(params.langnum)]

    return mapping, embedding, discriminators
