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
import scipy
import scipy.linalg
import torch
# from torch.autograd import Variable
from torch.nn import functional as F

from .utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from .utils import clip_parameters
from .dico_builder import build_dictionary
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary


logger = getLogger()


class Trainer():
    """train class"""

    def __init__(self, generator, discriminator, params):
        """
        Initialize trainer script.
        """

        self.embs = generator.embs
        self.dicos = params.dicos
        self.generator = generator
        self.discriminator = discriminator
        self.params = params
        self.langnum = self.params.langnum
        self._dicos = [0]*(self.langnum-1)

        # optimizers
        if hasattr(params, 'gen_optimizer'):
            optim_fn, optim_params = get_optimizer(params.gen_optimizer)
            self.gen_optimizer = optim_fn(generator.parameters(), **optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_params)
        else:
            assert discriminator is None

        # best validation score
        self.prev_metric = -1e12
        self.best_valid_metric = -1e12

        self.decrease_lr = False

    def get_dis_xy(self):
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
        ids = [0]*langnum
        for i in range(langnum):
            if self.params.test:
                ids[i] = torch.arange(0, bs, dtype=torch.int64)
            elif rv and i == langnum-1:
                ids[i] = torch.LongTensor(bs).random_(rv)
            else:
                ids[i] = torch.LongTensor(bs).random_(mf)
        if self.params.cuda:
            for i in range(self.langnum):
                ids[i] = ids[i].cuda()

        # get word embeddings
        embs = [0]*langnum
        for i in range(langnum-1):
            embs[i] = self.generator(self.embs[i](ids[i]).detach(), i)
        embs[-1] = self.embs[-1](ids[-1])
        if not rv:
            embs[-1] = embs[-1].detach()

        # if self.params.test:
            # logger.info('mean of absolute value of mapping %i is %.10f', 0, torch.mean(torch.abs(self.generator.mappings[1].weight)))
            # if isinstance(self.embs[2].weight.grad, torch.Tensor):
                # logger.info(self.embs[2].weight.grad.size())
                # logger.info(self.embs[2].weight.grad)

        # input / target
        x = torch.cat(embs, 0)

        # cross_entropyの場合
        y = torch.zeros((langnum * bs, langnum), dtype=torch.float32)
        for i in range(langnum):
            y[i*bs:(i+1)*bs, i] = 1-self.params.dis_smooth

        y = y.cuda() if self.params.cuda else y

        return x, y

    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()

        # loss
        x, y = self.get_dis_xy()
        preds = self.discriminator(x.detach())
        if self.params.test:
            logger.info('dis_start')
            # logger.info(torch.exp(preds[:10]))
            # logger.info(self.generator.mappings[0].weight[0][:10])

        # cross_entropyの場合
        loss = torch.mean(torch.sum(-y*preds, dim=1))

        # check NaN
        if (loss != loss).detach().any():
            logger.error("NaN detected (discriminator)")
            sys.exit()

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.params.clip_grad)
        self.dis_optimizer.step()

        if self.params.test and isinstance(self.embs[2].weight.grad, torch.Tensor):
            logger.info('%.15f', torch.mean(torch.abs(self.embs[2].weight.detach()[0])))
            logger.info('%.15f', torch.mean(torch.abs(self.embs[2].weight.grad[0])))

        self.discriminator.eval()
        new_preds = self.discriminator(x.detach())
        new_loss = torch.mean(torch.sum(-y*new_preds, dim=1))
        stats['DIS_COSTS'].append(new_loss.detach().item())
        if self.params.test:
            logger.info('after_dis')
            # logger.info(torch.exp(new_preds[:10]))
            # logger.info(self.discriminator.layers[1].weight.grad[0][:10])
            # logger.info(self.generator.mappings[0].weight[0][:10])
            # logger.info('Discriminator loss %.4f', new_loss)

    def gen_step(self, stats):
        """
        Fooling discriminator training step.
        """
        self.discriminator.eval()

        # loss
        x, y = self.get_dis_xy()
        # logger.info(torch.mean(torch.norm(self.embs[-1].weight.detach(), dim=1)))
        preds = self.discriminator(x)
        if self.params.test:
            logger.info('gen_start')
        #     logger.info(torch.exp(preds[:10]))
        #     logger.info(self.generator.mappings[0].weight[0][:10])

        loss = torch.mean(torch.sum(-(self.params.entropy_coef/self.langnum-y)*preds, dim=1))

        # check NaN
        if (loss != loss).detach().any():
            logger.error("NaN detected (fool discriminator)")
            sys.exit()

        # optim
        self.gen_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.params.clip_grad)
        self.gen_optimizer.step()
        if self.params.test:
            logger.info('%.15f', torch.mean(torch.abs(self.embs[2].weight.detach()[0])))
            logger.info('%.15f', torch.mean(torch.abs(self.embs[2].weight.grad[0])))

        new_x, new_y = self.get_dis_xy()
        new_preds = self.discriminator(new_x.detach())
        new_loss = torch.mean(torch.sum(-new_y*new_preds, dim=1))
        stats['MAP_COSTS'].append(new_loss.detach().item())
        if self.params.test:
            logger.info('after_gen')
        #     logger.info(torch.exp(new_preds[:10]))
        #     logger.info(self.generator.mappings[0].weight.grad[0][:10])
        #     logger.info(self.generator.mappings[0].weight[0][:10])
        #     logger.info('Mapping loss %.4f', new_loss)
        self.generator.orthogonalize()
        if self.params.test:
            logger.info('orthogonalized')
        #     x, y = self.get_dis_xy()
        #     logger.info(torch.exp(self.discriminator(x.detach())[:10]))
        #     logger.info(self.generator.mappings[0].weight[0][:10])

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
        if self.params.cuda:
            self.dico = self.dico.cuda()

    def build_dictionary(self):
        """
        Build a dictionary from aligned embeddings.
        """
        self._dicos = [0]*(self.langnum-1)
        tgt_emb = self.embs[-1].weight.detach()
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

        for i in range(self.langnum - 1):
            src_emb = self.generator(self.embs[i].weight, i).detach()
            src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
            self._dicos[i] = build_dictionary(src_emb, tgt_emb, self.params)

    def procrustes(self):
        """
        Find the best orthogonal matrix generator using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        for i in range(self.langnum - 1):
            A = self.embs[i].weight.detach()[self._dicos[i][:, 0]]
            B = self.embs[-1].weight.detach()[self._dicos[i][:, 1]]
            W = self.generator.mappings[i].weight.detach()
            M = B.transpose(0, 1).mm(A).cpu().numpy()
            U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
            W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.gen_optimizer:
            return
        old_lr = self.gen_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f", old_lr, new_lr)
            self.gen_optimizer.param_groups[0]['lr'] = new_lr
        if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            if to_log[metric] < self.prev_metric:
                logger.info("Validation metric is smaller than the previous one: %.5f vs %.5f", to_log[metric], self.prev_metric)
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                old_lr = self.gen_optimizer.param_groups[0]['lr']
                self.gen_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                logger.info("Shrinking the learning rate: %.5f -> %.5f", old_lr, self.gen_optimizer.param_groups[0]['lr'])
                self.decrease_lr = True
                self.params.epoch_size = int(self.params.epoch_size * self.params.lr_shrink)
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
            logger.info('* Best value for "%s": %.5f', metric, to_log[metric])
            # save the generator

            for i in range(self.langnum-1):
                W = self.generator.mappings[i].weight.detach().cpu().numpy()
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
            W = self.generator.mappings[i].weight.detach()
            assert to_reload.size() == W.size()
            W.copy_(to_reload.type_as(W))

    # def export(self):
    #     """
    #     Export embeddings.
    #     """
    #     params = self.params

    #     # load all embeddings
    #     logger.info("Reloading all embeddings for generator ...")
    #     embs = [0]*params.langnum
    #     for i in range(params.langnum):
    #         params.dicos[i], embs[i] = load_embeddings(params, i, full_vocab=True)

    #     # apply same normalization as during training
    #     for i in range(params.langnum):
    #         normalize_embeddings(embs[i], params.normalize_embeddings, mean=params.means[i])
    #     # normalize_embeddings(tgt_emb, params.normalize_embeddings, mean=params.tgt_mean)

    #     # map source embeddings to the target space
    #     bs = 4096
    #     logger.info("Map source embeddings to the target space ...")
    #     for j in range(params.langnum-1):
    #         for i, k in enumerate(range(0, len(embs[j]), bs)):
    #             with torch.no_grad():
    #                 x = embs[j][k:k + bs].cuda() if params.cuda else embs[j][k:k + bs]
    #             embs[j][k:k + bs] = self.generator[j](x).detach().cpu()

    #     # write embeddings to the disk
    #     export_embeddings(embs, params)
