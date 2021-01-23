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
    
    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.emb_dim = params.emb_dim
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        self.models = nn.ModuleList()
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
            self.models.append(nn.Sequential(*layers))

    def forward(self, x, i):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.models[i](x).view(-1)

class Mapping(nn.Module):
    """mapping and embeddings"""

    def __init__(self, params):
        super(Mapping, self).__init__()

        self.map_beta = params.map_beta
        self.langnum = params.langnum

        self.models = nn.ModuleList([nn.Linear(params.emb_dim, params.emb_dim, bias=False) for _ in range(self.langnum-1)])
        if getattr(params, 'map_id_init', True):
            for i in range(self.langnum-1):
                # self.models[i].weight.data = torch.diag(torch.ones(self.emb_dim))
                self.models[i].weight.data = torch.diag(torch.ones(params.emb_dim))

    def forward(self, x, i):
        """map into target space"""
        return self.models[i](x) if 0 <= i < self.langnum-1 else x

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        beta = self.map_beta
        for i in range(self.langnum-1):
            W = self.models[i].weight.detach()
            self.models[i].weight.data = (1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W))

class Embedding(nn.Module):
    """mapping and embeddings"""

    def __init__(self, params):
        super(Embedding, self).__init__()

        self.emb_dim = params.emb_dim
        self.langnum = params.langnum

        dicos, _embs = [0]*params.langnum, [0]*params.langnum
        for i in range(params.langnum):
            if i == params.langnum-1 and params.random_vocab:
                dicos[-1] = [0]*params.random_vocab
                _embs[-1] = torch.randn(params.random_vocab, params.emb_dim) / (params.emb_dim**.5)
                if params.emb_init == 'norm_mean':
                    norm_mean = torch.mean(torch.cat([torch.norm(_embs[l], dim=1, keepdim=True).expand_as(_embs[l]) for l in range(params.langnum-1)]).view(params.langnum-1, -1, params.emb_dim), dim=0)
                    _embs[-1] *= norm_mean[:params.random_vocab] / torch.mean(norm_mean[:params.random_vocab]) * params.emb_norm
                elif params.emb_init == 'en_mean':
                    norm_mean = torch.norm(_embs[0], dim=1, keepdim=True).expand_as(_embs[0])[:params.random_vocab]
                    _embs[-1] *= norm_mean[:params.random_vocab]
                elif params.emb_init == 'lang_mean':
                    norm_mean = torch.mean(torch.cat([torch.norm(_embs[l], dim=1, keepdim=True).expand_as(_embs[l]) for l in range(params.langnum-1)]).view(params.langnum-1, -1, params.emb_dim), dim=0)
                    _embs[-1] = torch.mean(torch.cat([_embs[l] for l in range(params.langnum-1)]).view(params.langnum-1, -1, params.emb_dim), dim=0)[:params.random_vocab]
                    _embs[-1].div_(_embs[-1].norm(2, 1, keepdim=True).expand_as(_embs[-1]))
                    _embs[-1] *= norm_mean[:params.random_vocab]
                elif params.emb_init == 'load':
                    _embs[-1] = torch.load(params.emb_file).to('cpu')
                else:
                    _embs[-1] *= params.emb_norm
            else:
                dicos[i], _embs[i] = load_embeddings(params, i)
        self.embs = nn.ModuleList([nn.Embedding(len(dicos[i]), params.emb_dim, sparse=False) for i in range(self.langnum)])
        for i in range(self.langnum):
            self.embs[i].weight.data = _embs[i]
            if i == self.langnum-1 and params.learnable:
                self.embs[i].weight.requires_grad = True
            else:
                self.embs[i].weight.requires_grad = False
        params.dicos = dicos

def build_model(params, with_dis=True):
    """
    Build all components of the model.
    """
    mapping = Mapping(params).to(params.device)
    embedding = Embedding(params).to(params.device)

    # discriminator
    discriminators = Discriminator(params).to(params.device) if with_dis else None

    # normalize embeddings
    params.means = [normalize_embeddings(embedding.embs[i].weight.detach(), params.normalize_embeddings) for i in range(params.langnum)]

    return mapping, embedding, discriminators
