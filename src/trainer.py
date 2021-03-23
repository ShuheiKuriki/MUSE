"""train"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
from logging import getLogger
import itertools
import scipy
import scipy.linalg
import random
import torch
# from torch.autograd import Variable
from torch.nn import functional as F
from copy import deepcopy
import numpy as np

from .utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from .utils import clip_parameters
from .dico_builder import build_dictionary
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary


logger = getLogger()


class Trainer():
    """train class"""

    def __init__(self, mapping, embedding, discriminator, params):
        """
        Initialize trainer script.
        """

        self.embedding = embedding
        self.embs = embedding.embs
        self.vocabs = params.vocabs
        self.mapping = mapping
        self.discriminator = discriminator
        self.params = params
        self.langnum = self.params.langnum
        self.dicos = [[0]*self.langnum for _ in range(self.langnum)]

        # optimizers
        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            self.map_optimizer = optim_fn(mapping.parameters(), **optim_params)
        if hasattr(params, 'emb_optimizer'):
            optim_fn, optim_params = get_optimizer(params.emb_optimizer)
            self.emb_optimizer = optim_fn(embedding.parameters(), **optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_params)
        else:
            assert discriminator is None
        if hasattr(params, 'ref_optimizer'):
            optim_fn, optim_params = get_optimizer(params.ref_optimizer)
            self.ref_optimizer = optim_fn(mapping.parameters(), **optim_params)
        if hasattr(params, 'emb_ref_optimizer'):
            optim_fn, optim_params = get_optimizer(params.emb_ref_optimizer)
            self.emb_ref_optimizer = optim_fn(embedding.parameters(), **optim_params)

        if params.test:
            logger.info('mapping')
            for param in mapping.parameters():
                logger.info(param.requires_grad)
            # logger.info(mapping.linear[0].weight.requires_grad)
            logger.info('embedding')
            for param in embedding.parameters():
                logger.info(param.requires_grad)
            # logger.info(embedding.embs[-1].weight.requires_grad)
            logger.info('discriminator')
            for param in discriminator.parameters():
                logger.info(param.requires_grad)
            # logger.info(discriminator.models[0][1].weight.requires_grad)

        # best validation score
        self.prev_metric = -1e12
        self.best_valid_metric = -1e12

        self.decrease_lr = False

    def get_dis_xy(self, i, j, mode='dis'):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        uv = self.params.univ_vocab
        langnum = self.langnum
        assert mf <= min(map(len, self.vocabs[:-1]))

        # get ids
        if self.params.test:
            src_ids = torch.arange(0, bs, dtype=torch.int64).to(self.params.device)
            tgt_ids = torch.arange(0, bs, dtype=torch.int64).to(self.params.device)
        else:
            if uv > 0 and i == langnum-1:
                src_ids = torch.LongTensor(bs).random_(uv).to(self.params.device)
            else:
                src_ids = torch.LongTensor(bs).random_(mf).to(self.params.device)
            if uv > 0 and j == langnum-1:
                tgt_ids = torch.LongTensor(bs).random_(uv).to(self.params.device)
            else:
                tgt_ids = torch.LongTensor(bs).random_(mf).to(self.params.device)

        # get word embeddings
        if mode == 'dis':
            src_emb = self.embs[i](src_ids).detach()
            tgt_emb = self.embs[j](tgt_ids).detach()
        else:
            src_emb = self.embs[i](src_ids)
            tgt_emb = self.embs[j](tgt_ids)

        src_emb = self.mapping(src_emb, i, j)

        # if self.params.test:
            # logger.info('mean of absolute value of mapping %i is %.10f', 0, torch.mean(torch.abs(self.mapping.linear[1].weight)))
            # if isinstance(self.embs[2].weight.grad, torch.Tensor):
                # logger.info(self.embs[2].weight.grad.size())
                # logger.info(self.embs[2].weight.grad)

        # input / target
        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.FloatTensor(2 * bs).zero_()

        # 0 indicates real (lang2) samples
        y[:bs] = 1 - self.params.dis_smooth
        y[bs:] = self.params.dis_smooth
        y = y.to(self.params.device)

        return x, y

    def get_refine_xy(self, i, j):
        """
        Get input batch / output target for MPSR.
        """
        # select random word IDs
        bs = self.params.batch_size
        dico = self.dicos[i][j]
        indices = torch.from_numpy(np.random.randint(0, len(dico), bs)).to(self.params.device)
        dico = dico.index_select(0, indices)
        src_ids = dico[:, 0].to(self.params.device)
        tgt_ids = dico[:, 1].to(self.params.device)

        # get word embeddings
        x = self.mapping(self.embs[i](src_ids), i, j)
        y = self.embs[j](tgt_ids)

        return x, y

    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()

        # loss
        loss = 0
        # for each target language
        for j in range(self.langnum):
            # random select a source language
            i = random.randint(0, self.langnum-1)
            x, y = self.get_dis_xy(i, j, mode='dis')
            preds = self.discriminator(x.detach(), j)
            loss += F.binary_cross_entropy(preds, y)

        if self.params.test:
            logger.info('dis_start')
            logger.info(self.discriminator.models[0][1].weight[0][:10])
            logger.info(self.discriminator.models[1][1].weight[0][:10])

        # check NaN
        if (loss != loss).detach().any():
            logger.error("NaN detected (discriminator)")
            sys.exit()

        stats['DIS_COSTS'].append(loss.detach().item())

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.params.clip_grad)

        if self.params.test:
            logger.info('after_dis')
            logger.info(self.discriminator.models[0][1].weight[0][:10])
            logger.info(self.discriminator.models[1][1].weight[0][:10])

    def gen_step(self):
        """
        Fooling discriminator training step.
        """
        self.discriminator.eval()

        # loss
        loss = words = 0
        # for each source language
        for i in range(self.langnum):
            # randomly select a target language
            j = random.randint(0, self.langnum-1)
            x, y = self.get_dis_xy(i, j, mode='gen')
            preds = self.discriminator(x, j)
            loss += F.binary_cross_entropy(preds, 1-y)
            words += 2 * self.params.batch_size

        # check NaN
        if (loss != loss).detach().any():
            logger.error("NaN detected (fool discriminator)")
            sys.exit()

        # optim
        self.map_optimizer.zero_grad()
        if self.params.learnable: self.emb_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.mapping.parameters(), self.params.clip_grad)
        self.map_optimizer.step()
        if self.params.learnable:
            # torch.nn.utils.clip_grad_norm_(self.embedding.parameters(), self.params.clip_grad)
            self.emb_optimizer.step()
        self.mapping.orthogonalize()

        if self.params.test:
            logger.info('after_gen')
            # print(torch.norm(self.discriminator.layers[1].weight))
            logger.info(torch.norm(self.discriminator.models[0][1].weight[0]))
            logger.info(torch.norm(self.discriminator.models[1][1].weight[0]))

        return words

    def refine_step(self, stats):
        """mpsr step"""
        loss = words = 0
        for i in range(self.langnum-1):
            if self.params.langs[-1] == 'random':
                j = random.randint(0, self.langnum-2)
            else:
                j = random.randint(0, self.langnum-1)
            x, y = self.get_refine_xy(i, j)
            loss += F.mse_loss(x, y)
            words += 2 * self.params.batch_size
        for _ in range(self.params.ref_tgt):
            if self.params.langs[-1] == 'random':
                j = random.randint(0, self.langnum-2)
            else:
                j = random.randint(0, self.langnum-1)
            x, y = self.get_refine_xy(self.langnum-1, j)
            loss += F.mse_loss(x, y)
            words += 2 * self.params.batch_size
        # check NaN
        if (loss != loss).any():
            logger.error("NaN detected (fool discriminator)")
            sys.exit()

        stats['REFINE_COSTS'].append(loss.item())
        # optim
        self.ref_optimizer.zero_grad()
        if self.params.learnable: self.emb_ref_optimizer.zero_grad()
        loss.backward()
        self.ref_optimizer.step()
        if self.params.learnable: self.emb_ref_optimizer.step()
        self.mapping.orthogonalize()

        return words

    def load_training_dico(self, dico_train):
        """
        Load training dictionary.
        """
        for i in range(self.params.langnum):
            for j in range(self.params.langnum):
                if i == j:
                    idx = torch.arange(self.params.dico_max_rank).long().view(self.params.dico_max_rank, 1)
                    self.dicos[i][j] = torch.cat([idx, idx], dim=1).to(self.params.device)
                else:
                    word2id1 = self.vocabs[i].word2id
                    word2id2 = self.vocabs[j].word2id

                    # identical character strings
                    if dico_train == "identical_char":
                        self.dicos[i][j] = load_identical_char_dico(word2id1, word2id2)
                    # use one of the provided dictionary
                    elif dico_train == "default":
                        filename = f'{self.params.langs[i]}-{self.params.langs[j]}.0-5000.txt'
                        self.dicos[i][j] = load_dictionary(os.path.join(DIC_EVAL_PATH, filename), word2id1, word2id2)
                    # dictionary provided by the user
                    else:
                        self.dicos[i][j] = load_dictionary(dico_train, word2id1, word2id2)

                    # cuda
                    self.dicos[i][j] = self.dicos[i][j].to(self.params.device)

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        embs = [self.mapping(self.embs[i].weight, i).detach() for i in range(self.langnum)]
        embs = [embs[i] / embs[i].norm(2, 1, keepdim=True).expand_as(embs[i]) for i in range(self.langnum)]

        _params = deepcopy(self.params)
        idx = torch.arange(_params.dico_max_rank).long().view(_params.dico_max_rank, 1)
        for i in range(self.langnum-1):
            for j in range(self.langnum-1):
                if i != j: logger.info('%s %s', _params.langs[i], _params.langs[j])
                if i < j:
                    logger.info("Building the train dictionary between %s and %s", _params.langs[i], _params.langs[j])
                    self.dicos[i][j] = build_dictionary(embs[i], embs[j], _params)
                elif i > j:
                    self.dicos[i][j] = self.dicos[j][i][:, [1, 0]]
                else:
                    self.dicos[i][j] = torch.cat([idx, idx], dim=1).to(_params.device)
        if _params.langs[-1] == 'random':
            _params.dico_max_rank = max(15000, _params.univ_vocab)
        for i in range(self.langnum-1):
            logger.info('%s %s', _params.langs[i], _params.langs[-1])
            self.dicos[i][-1] = build_dictionary(embs[i], embs[-1], _params)
        for j in range(self.langnum-1):
            logger.info('%s %s', _params.langs[-1], _params.langs[j])
            self.dicos[-1][j] = self.dicos[j][-1][:, [1, 0]]
        idx = torch.arange(_params.dico_max_rank).long().view(_params.dico_max_rank, 1)
        self.dicos[-1][-1] = torch.cat([idx, idx], dim=1).to(_params.device)

    def procrustes(self):
        """
        Find the best orthogonal matrix generator using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        for i in range(self.langnum - 1):
            A = self.embs[i].weight.detach()[self.dicos[i][-1][:, 0]]
            B = self.embs[-1].weight.detach()[self.dicos[i][-1][:, 1]]
            W = self.mapping.linear[i].weight.detach()
            M = B.transpose(0, 1).mm(A).cpu().numpy()
            U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
            W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    def update_lr(self, to_log, metric, modes='map'):
        """
        Update learning rate when using SGD.
        """
        optimizers = []
        if 'map' in modes and self.params.map_optimizer[:3] == 'sgd':
            optimizers.append(('map', self.map_optimizer))
        if 'emb' in modes and self.params.emb_optimizer[:3] == 'sgd':
            optimizers.append(('emb', self.emb_optimizer))
        if 'ref' in modes and self.params.ref_optimizer[:3] == 'sgd':
            optimizers.append(('ref', self.ref_optimizer))
        if 'emb_ref' in modes and self.params.emb_ref_optimizer[:3] == 'sgd':
            optimizers.append(('emb_ref', self.emb_ref_optimizer))

        if to_log[metric] < self.prev_metric:
            update = True
            logger.info("Validation metric is smaller than the previous one: %.5f vs %.5f", to_log[metric], self.prev_metric)
        else:
            update = False
            logger.info("The validation metric is getting better %.5f → %.5f", self.prev_metric, to_log[metric])
        self.prev_metric = to_log[metric]
        for mode, optimizer in optimizers:
            old_lr = optimizer.param_groups[0]['lr']
            new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
            if new_lr < old_lr:
                logger.info("Decreasing learning rate: %s %.8f -> %.8f ", mode, old_lr, new_lr)
                optimizer.param_groups[0]['lr'] = new_lr
            if self.params.lr_shrink == 1 or not update: continue
            # decrease the learning rate, only if this is the
            # second time the validation metric decreases
            old_lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
            logger.info("Shrinking the learning rate: %s %.5f -> %.5f", mode, old_lr, old_lr*self.params.lr_shrink)

    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best generator for the given validation criterion
        if to_log[metric] > self.best_valid_metric:
            # new best generator
            self.best_valid_metric = to_log[metric]
            self.best_epoch = to_log["n_epoch"]
            self.best_tgt_norm = to_log["tgt_norm"]
            logger.info('* Best value for "%s": %.5f', metric, to_log[metric])
            # save the generator

            for i in range(self.langnum-1):
                path = os.path.join(self.params.exp_path, 'best_mapping{}.pth'.format(i+1))
                logger.info('* Saving the generator to %s ...', path)
                torch.save(self.mapping.linear[i].weight.detach().cpu(), path)

            if self.params.learnable:
                path = os.path.join(self.params.exp_path, 'vectors-%s.pth' % self.params.langs[-1])
                logger.info('Writing universal embeddings to %s ...', path)
                torch.save(self.embs[-1].weight.detach().cpu(), path)

    def reload_best(self):
        """
        Reload the best generator.
        """
        for i in range(self.langnum-1):
            path = os.path.join(self.params.exp_path, 'best_mapping{}.pth'.format(i+1))
            logger.info('* Reloading the best model from %s ...', path)
            # reload the model
            assert os.path.isfile(path)
            self.mapping.linear[i].weight.data = torch.load(path).to(self.params.device)

        if self.params.learnable:
            path = os.path.join(self.params.exp_path, 'vectors-%s.pth' % self.params.langs[-1])
            logger.info('* Reloading the best universal embedding from %s ...', path)
            # reload the model
            assert os.path.isfile(path)
            self.embs[-1].weight.data = torch.load(path).to(self.params.device)

    def reload(self, folder):
        """
        Reload mappibgs from given folder.
        """
        for i in range(self.langnum-1):
            path = os.path.join(folder, 'best_mapping{}.pth'.format(i+1))
            logger.info('* Reloading the model from %s ...', path)
            # reload the model
            assert os.path.isfile(path)
            to_reload = torch.from_numpy(torch.load(path))
            W = self.mapping.linear[i].weight.detach()
            assert to_reload.size() == W.size()
            W.copy_(to_reload.type_as(W))

    def export(self):
        """
        Export embeddings.
        """
        params = self.params

        # load all embeddings
        logger.info("Reloading all embeddings for generator ...")
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
                x = embs[j][k:k + bs].to(self.params.device)
                embs[j][k:k + bs] = self.mapping.linear[j](x).detach().cpu()

        # write embeddings to the disk
        export_embeddings(embs, params)
