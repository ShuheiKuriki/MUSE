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
import numpy as np

from .utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from .utils import clip_parameters
from .dico_builder import build_dictionary
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary


logger = getLogger()


class Trainer():
    """train class"""

    def __init__(self, mapping, embedding, discriminators, params):
        """
        Initialize trainer script.
        """

        self.embedding = embedding
        self.embs = embedding.embs
        self.dicos = params.dicos
        self.mapping = mapping
        self.discriminators = discriminators
        self.params = params
        self.langnum = self.params.langnum
        self._dicos = [0]*(self.langnum-1)

        # optimizers
        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            self.map_optimizer = optim_fn(mapping.parameters(), **optim_params)
        if hasattr(params, 'emb_optimizer'):
            optim_fn, optim_params = get_optimizer(params.emb_optimizer)
            self.emb_optimizer = optim_fn(embedding.parameters(), **optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            # self.dis_optimizer = optim_fn(itertools.chain(*[d.parameters() for d in discriminators.models]), **optim_params)
            self.dis_optimizer = optim_fn(discriminators.parameters(), **optim_params)
        else:
            assert discriminators is None
        if hasattr(params, 'ref_optimizer'):
            optim_fn, optim_params = get_optimizer(params.ref_optimizer)
            self.ref_optimizer = optim_fn(mapping.parameters(), **optim_params)
        if hasattr(params, 'emb_ref_optimizer'):
            optim_fn, optim_params = get_optimizer(params.emb_ref_optimizer)
            self.emb_ref_optimizer = optim_fn(embedding.parameters(), **optim_params)

        if params.test:
            logger.info('mapping')
            for param in mapping.parameters():
                logger.info(param.size())
            logger.info(mapping.models[0].weight.requires_grad)
            logger.info('embedding')
            for param in embedding.parameters():
                logger.info(param.size())
            logger.info(embedding.embs[0].weight.requires_grad)
            logger.info('discriminator')
            for param in discriminators.parameters():
                logger.info(param.size())
            logger.info(discriminators.models[0][1].weight.requires_grad)

        # best validation score
        self.prev_metric = -1e12
        self.best_valid_metric = -1e12

        self.decrease_lr = False

    def get_dis_xy(self, i, j):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        rv = self.params.random_vocab
        langnum = self.langnum
        assert mf <= min(map(len, self.dicos))

        # get ids
        if self.params.test:
            # src_ids = torch.arange(0, bs, dtype=torch.int64).to(self.params.device)
            # tgt_ids = torch.arange(0, bs, dtype=torch.int64).to(self.params.device)
            src_ids = torch.LongTensor(bs).random_(mf).to(self.params.device)
            tgt_ids = torch.LongTensor(bs).random_(mf).to(self.params.device)
        elif rv and i == langnum-1:
            src_ids = torch.LongTensor(bs).random_(rv).to(self.params.device)
            tgt_ids = torch.LongTensor(bs).random_(rv).to(self.params.device)
        else:
            src_ids = torch.LongTensor(bs).random_(mf).to(self.params.device)
            tgt_ids = torch.LongTensor(bs).random_(mf).to(self.params.device)

        # get word embeddings
        src_emb = self.mapping(self.mapping(self.embs[i](src_ids), i), j, rev=True)
        tgt_emb = self.embs[j](tgt_ids)

        # if self.params.test:
            # logger.info('mean of absolute value of mapping %i is %.10f', 0, torch.mean(torch.abs(self.mapping.models[1].weight)))
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
        dico = self._dicos[i][j]
        indices = torch.from_numpy(np.random.randint(0, len(dico), bs)).to(self.params.device)
        dico = dico.index_select(0, indices)
        src_ids = dico[:, 0].to(self.params.device)
        tgt_ids = dico[:, 1].to(self.params.device)

        # get word embeddings
        x = self.mapping(self.mapping(self.embs[i](src_ids), i), j, rev=True)
        y = self.embs[j](tgt_ids)

        return x, y

    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminators.train()

        # loss
        loss = 0
        # for each target language
        for j in range(self.langnum):
            # random select a source language
            i = random.choice(list(range(self.langnum)))

            x, y = self.get_dis_xy(i, j)
            preds = self.discriminators(x.detach(), j)
            loss += F.binary_cross_entropy(preds, y)

        if self.params.test:
            logger.info('dis_start')
            logger.info(self.discriminators.models[0][1].weight[0][:10])
            logger.info(self.discriminators.models[1][1].weight[0][:10])
            # logger.info(torch.norm(self.discriminators.models[0][4].weight[0]))
            # logger.info(torch.norm(self.discriminators.models[1][4].weight[0]))
            # logger.info(preds[:10])
            # logger.info(self.mapping.models[0].weight[0][:10])

        # check NaN
        if (loss != loss).detach().any():
            logger.error("NaN detected (discriminator)")
            sys.exit()

        stats['DIS_COSTS'].append(loss.detach().item())

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        # torch.nn.utils.clip_grad_norm_(self.discriminators.parameters(), self.params.clip_grad)

        if self.params.test:
            logger.info('after_dis')
            # for i in range(self.langnum):
                # logger.info('%.15f', torch.mean(torch.norm(self.embs[i].weight.detach()[0])))
                # logger.info('%.15f', torch.mean(torch.norm(self.embs[i].weight.grad[0])))
            # logger.info(torch.exp(new_preds[:10]))
            # logger.info(self.discriminators.models[0][1].weight.grad[0][:10])
            logger.info(self.discriminators.models[0][1].weight[0][:10])
            logger.info(self.discriminators.models[1][1].weight[0][:10])
            # logger.info(torch.norm(self.discriminators.models[0][4].weight[0]))
            # logger.info(torch.norm(self.discriminators.models[1][4].weight[0]))
            # print(torch.norm(self.discriminator.layers[1].weight))
            # logger.info(self.mapping.models[0].weight[0][:10])
            # logger.info('Discriminator loss %.4f', new_loss)

    def gen_step(self, stats, mode='map'):
        """
        Fooling discriminator training step.
        """
        self.discriminators.eval()

        # loss
        loss = 0
        if mode == 'map':
            # for each source language
            for i in range(self.langnum):
                # random select a target language
                j = random.choice(list(range(self.langnum)))
                x, y = self.get_dis_xy(i, j)
                preds = self.discriminators(x, j)
                loss += F.binary_cross_entropy(preds, 1-y)
        elif mode == 'emb':
            i = random.choice(list(range(self.langnum-1)))
            x, y = self.get_dis_xy(i, self.langnum-1)
            preds = self.discriminators(x, self.langnum-1)
            loss += F.binary_cross_entropy(preds, 1-y)

        # check NaN
        if (loss != loss).detach().any():
            logger.error("NaN detected (fool discriminator)")
            sys.exit()

        # optim
        if mode == 'map':
            self.map_optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.mapping.parameters(), self.params.clip_grad)
            self.map_optimizer.step()
            self.mapping.orthogonalize()
        elif mode == 'emb':
            self.emb_optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.embedding.parameters(), self.params.clip_grad)
            self.emb_optimizer.step()

        if self.params.test:
            logger.info('after_%s', mode)
            # print(torch.norm(self.discriminator.layers[1].weight))
            logger.info(torch.norm(self.discriminators.models[0][1].weight[0]))
            logger.info(torch.norm(self.discriminators.models[1][1].weight[0]))

        return self.langnum * self.params.batch_size

    def refine_step(self, stats, mode='map'):
        # loss
        loss = 0
        if mode == 'map':
            for i in range(self.langnum):
                j = random.choice(list(range(self.langnum)))
                x, y = self.get_refine_xy(i, j)
                loss += F.mse_loss(x, y)
        elif mode == 'emb':
            i = random.choice(list(range(self.langnum-1)))
            x, y = self.get_refine_xy(i, self.langnum-1)
            loss += F.mse_loss(x, y)
        # check NaN
        if (loss != loss).any():
            logger.error("NaN detected (fool discriminator)")
            sys.exit()

        stats['REFINE_COSTS'].append(loss.item())
        # optim
        if mode == 'map':
            self.ref_optimizer.zero_grad()
            loss.backward()
            self.ref_optimizer.step()
            self.mapping.orthogonalize()
        elif mode == 'emb':
            self.emb_ref_optimizer.zero_grad()
            loss.backward()
            self.emb_ref_optimizer.step()

        return self.langnum * self.params.batch_size

    def load_training_dico(self, dico_train):
        """
        Load training dictionary.
        """
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

        # identical character strings
        if dico_train == "identical_char":
            self.dico = load_identical_char_dico(word2id1, word2id2)
        # use one of the provided dictionary
        elif dico_train == "default":
            filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
            self.dico = load_dictionary(
                os.path.join(DIC_EVAL_PATH, filename),
                word2id1, word2id2
            )
        # dictionary provided by the user
        else:
            self.dico = load_dictionary(dico_train, word2id1, word2id2)

        # cuda
        self.dico = self.dico.to(self.params.device)

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        self._dicos = [[0]*self.langnum for _ in range(self.langnum)]
        embs = [self.mapping(self.embs[i].weight, i).detach() for i in range(self.langnum)]
        embs = [embs[i] / embs[i].norm(2, 1, keepdim=True).expand_as(embs[i]) for i in range(self.langnum)]

        for i in range(self.langnum):
            for j in range(self.langnum):
                if i < j:
                    self._dicos[i][j] = build_dictionary(embs[i], embs[j], self.params)
                elif i > j:
                    self._dicos[i][j] = self._dicos[j][i][:, [1, 0]]
                else:
                    idx = torch.arange(self.params.dico_max_rank).long().view(self.params.dico_max_rank, 1)
                    self._dicos[i][j] = torch.cat([idx, idx], dim=1).to(self.params.device)

    def procrustes(self):
        """
        Find the best orthogonal matrix generator using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        for i in range(self.langnum - 1):
            A = self.embs[i].weight.detach()[self._dicos[i][:, 0]]
            B = self.embs[-1].weight.detach()[self._dicos[i][:, 1]]
            W = self.mapping.models[i].weight.detach()
            M = B.transpose(0, 1).mm(A).cpu().numpy()
            U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
            W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    def procrustes2(self, i):
        """
        Find the best orthogonal matrix generator using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """

        self._dicos = [0]*(self.langnum-1)
        tgt_emb = self.mapping(self.embs[i].weight, i).detach()
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        for j in range(self.langnum-1):
            if i == j: continue
            src_emb = self.mapping(self.embs[j].weight, j).detach()
            src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
            self._dicos[j] = build_dictionary(src_emb, tgt_emb, self.params)

            A = self.embs[j].weight.detach()[self._dicos[j][:, 0]]
            B = self.mapping(self.embs[i].weight, i).detach()[self._dicos[j][:, 1]]
            W = self.mapping.models[j].weight.detach()
            M = B.transpose(0, 1).mm(A).cpu().numpy()
            U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
            W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    def update_lr(self, to_log, metric, mode='map'):
        """
        Update learning rate when using SGD.
        """
        if mode == 'map':
            optimizer = self.map_optimizer
        elif mode == 'emb':
            optimizer = self.emb_optimizer
        elif mode == 'ref':
            optimizer = self.ref_optimizer

        if mode == 'map' and 'sgd' not in self.params.map_optimizer: return
        if mode == 'emb' and 'sgd' not in self.params.emb_optimizer: return
        if mode == 'ref' and 'sgd' not in self.params.ref_optimizer: return

        old_lr = optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %s %.8f -> %.8f ", mode, old_lr, new_lr)
            optimizer.param_groups[0]['lr'] = new_lr
        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.prev_metric:
                logger.info("Validation metric is smaller than the previous one: %.5f vs %.5f", to_log[metric], self.prev_metric)
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                old_lr = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                logger.info("Shrinking the learning rate: %s %.5f -> %.5f", mode, old_lr, old_lr*self.params.lr_shrink)
                self.decrease_lr = True
            else:
                logger.info("The validation metric is getting better")
            self.prev_metric = to_log[metric]

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
                W = self.mapping.models[i].weight.detach().cpu().numpy()
                path = os.path.join(self.params.exp_path, 'best_mapping{}.pth'.format(i+1))
                logger.info('* Saving the generator to %s ...', path)
                torch.save(W, path)

    def reload_best(self):
        """
        Reload the best generator.
        """
        for i in range(self.langnum-1):
            path = os.path.join(self.params.exp_path, 'best_mapping{}.pth'.format(i+1))
            logger.info('* Reloading the best model from %s ...', path)
            # reload the model
            assert os.path.isfile(path)
            to_reload = torch.from_numpy(torch.load(path))
            W = self.mapping.models[i].weight.detach()
            assert to_reload.size() == W.size()
            W.copy_(to_reload.type_as(W))

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
            W = self.mapping.models[i].weight.detach()
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
                embs[j][k:k + bs] = self.mapping.models[j](x).detach().cpu()

        # write embeddings to the disk
        export_embeddings(embs, params)
