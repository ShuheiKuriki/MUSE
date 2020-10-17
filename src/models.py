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
            # output_dim = params.langnum if i == self.dis_layers else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """calculate forward"""
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        # output = self.layers(x).view(-1, self.params.langnum)
        output = self.layers(x).view(-1)
        # print(output)
        return output

class Generator(nn.Module):
    """mapping"""

    def __init__(self, params):
        super(Generator, self).__init__()

        self.emb_dim = params.emb_dim
        self.map_beta = params.map_beta
        self.langnum = params.langnum

        self.mappings = [nn.Linear(params.emb_dim, params.emb_dim, bias=False) for _ in range(self.langnum-1)]
        if getattr(params, 'map_id_init', True):
            for i in range(self.langnum-1):
                self.mappings[i].weight.data.copy_(torch.diag(torch.ones(self.emb_dim)))

    def forward(self, x, i):
        """map into target space"""
        assert x.dim() == 2 and x.size(1) == self.emb_dim
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
    
    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best mapping
            self.best_valid_metric = to_log[metric]
            logger.info('* Best value for "%s": %.5f', metric, to_log[metric])
            # save the mapping
            W = self.mapping.weight.detach().cpu().numpy()
            path = os.path.join(self.params.exp_path, 'best_mapping.pth')
            logger.info('* Saving the mapping to %s ...', path)
            torch.save(W, path)

    def reload_best(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params.exp_path, 'best_mapping.pth')
        logger.info('* Reloading the best model from %s ...', path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        W = self.mapping.weight.detach()
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))

    def export(self):
        """
        Export embeddings.
        """
        params = self.params

        # load all embeddings
        logger.info("Reloading all embeddings for mapping ...")
        embs = [0]*params.langnum
        for i in range(params.langnum):
            params.dicos[i], embs[i] = load_embeddings(params, i, full_vocab=True)

        # apply same normalization as during training
        for i in range(params.langnum):
            normalize_embeddings(embs[i], params.normalize_embeddings, mean=params.means[i])
        # normalize_embeddings(tgt_emb, params.normalize_embeddings, mean=params.tgt_mean)

        # map source embeddings to the target space
        bs = 4096
        logger.info("Map source embeddings to the target space ...")
        for j in range(params.langnum-1):
            for i, k in enumerate(range(0, len(embs[j]), bs)):
                with torch.no_grad():
                    x = embs[j][k:k + bs].cuda() if params.cuda else embs[j][k:k + bs]
                embs[j][k:k + bs] = self.mappings[j](x).detach().cpu()

        # write embeddings to the disk
        export_embeddings(embs, params)


def build_model(params, with_dis):
    """
    Build all components of the model.
    """
    # embeddings
    dicos, _embs = [0]*params.langnum, [0]*params.langnum
    for i in range(params.langnum):
        dicos[i], _embs[i] = load_embeddings(params, i)
    params.dicos = dicos
    embs = [nn.Embedding(len(dicos[i]), params.emb_dim, sparse=True) for i in range(params.langnum)]
    for i in range(params.langnum):
        embs[i].weight.data.copy_(_embs[i])

    # mapping
    # mappings = [nn.Linear(params.emb_dim, params.emb_dim, bias=False) for _ in range(params.langnum-1)]
    # if getattr(params, 'map_id_init', True):
        # for i in range(params.langnum-1):
            # mappings[i].weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))
    mappings = Generator(params)

    # discriminator
    discriminator = Discriminator(params) if with_dis else None

    # cuda
    if params.cuda:
        for i in range(params.langnum):
            embs[i].cuda()
        for i in range(params.langnum-1):
            mappings[i].cuda()
        if with_dis:
            discriminator.cuda()

    # normalize embeddings
    params.means = [normalize_embeddings(embs[i].weight.data, params.normalize_embeddings) for i in range(params.langnum)]

    return embs, mappings, discriminator
