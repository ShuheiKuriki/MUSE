"""モデル構築"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn

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
        layers.append(nn.Softmax())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1, self.params.langnum)


def build_model(params, with_dis):
    """
    Build all components of the model.
    """
    dicos, _embs = [0]*params.langnum, [0]*params.langnum
    for i in range(params.langnum):
        dicos[i], _embs[i] = load_embeddings(params, i)
        print(_embs[i].size())
    params.dicos = dicos
    embs = [nn.Embedding(len(dicos[i]), params.emb_dim, sparse=True) for i in range(params.langnum)]
    for i in range(params.langnum):
        embs[i].weight.data.copy_(_embs[i])

    # target embeddings
    # if params.tgt_lang:
    #     tgt_dico, _tgt_emb = load_embeddings(params, source=False)
    #     params.tgt_dico = tgt_dico
    #     tgt_emb = nn.Embedding(len(tgt_dico), params.emb_dim, sparse=True)
    #     tgt_emb.weight.data.copy_(_tgt_emb)
    # else:
    #     tgt_emb = None
    # if params.third_lang:
    #     third_dico, _third_emb = load_embeddings(params, source=False)
    #     params.third_dico = third_dico
    #     third_emb = nn.Embedding(len(third_dico), params.emb_dim, sparse=True)
    #     third_emb.weight.data.copy_(_third_emb)
    # else:
    #     third_emb = None

    # mapping
    mappings = [nn.Linear(params.emb_dim, params.emb_dim, bias=False) for _ in range(params.langnum-1)]
    if getattr(params, 'map_id_init', True):
        for i in range(params.langnum):
            mappings[i].weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))
    # tgt_mapping = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
    # if getattr(params, 'map_id_init', True):
    #     tgt_mapping.weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))

    # discriminator
    discriminator = Discriminator(params) if with_dis else None

    # cuda
    if params.cuda:
        for i in range(params.langnum):
            embs[i].cuda()
        for i in range(params.langnum-1):
            mappings[i].cuda()
        # if params.tgt_lang:
        #     tgt_emb.cuda()
        #     tgt_mapping.cuda()
        # if params.third_lang:
        #     third_emb.cuda()
        if with_dis:
            discriminator.cuda()

    # normalize embeddings
    params.means = [normalize_embeddings(embs[i].weight.data, params.normalize_embeddings) for i in range(params.langnum)]
    # for i in range(params.langnum):
        # params.src_mean = normalize_embeddings(embs[i].weight.data, params.normalize_embeddings)
    # if params.tgt_lang:
        # params.tgt_mean = normalize_embeddings(tgt_emb.weight.data, params.normalize_embeddings)
    # if params.third_lang:
        # params.third_mean = normalize_embeddings(third_emb.weight.data, params.normalize_embeddings)

    return embs, mappings, discriminator
