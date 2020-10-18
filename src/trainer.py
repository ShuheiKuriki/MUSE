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

    def __init__(self, embs, generator, discriminator, params):
        """
        Initialize trainer script.
        """

        self.embs = embs
        self.dicos = params.dicos
        self.generator = generator
        self.discriminator = discriminator
        self.params = params

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
        self.best_valid_metric = -1e12

        self.decrease_lr = False

    def get_dis_xy(self, volatile):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        langnum = self.params.langnum
        assert mf <= min(map(len, self.dicos))
        ids = [0]*langnum
        for i in range(langnum):
            ids[i] = torch.LongTensor(bs).random_(len(self.dicos[i]) if mf == 0 else mf)
            if self.params.cuda:
                ids[i] = ids[i].cuda()

        # get word embeddings
        embs = [0]*langnum
        if volatile:
            with torch.no_grad():
                for i in range(langnum):
                    embs[i] = self.embs[i](ids[i])
                for i in range(langnum-1):
                    embs[i] = self.generator(embs[i].detach(), i)
                # embs[-1] = embs[-1].detach()
        else:
            for i in range(langnum):
                embs[i] = self.embs[i](ids[i])
            for i in range(langnum-1):
                embs[i] = self.generator(embs[i].detach(), i)
            # embs[-1] = embs[-1].detach()

        # input / target
        x = torch.cat(embs, 0)
        # y = torch.zeros(langnum * bs, dtype=torch.int64)
        y = torch.FloatTensor(langnum * bs).zero_()
        for i in range(langnum):
            y[i*bs:(i+1)*bs] = 1-i
        y = y.cuda() if self.params.cuda else y

        return x, y

    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()
        self.generator.eval()

        # loss
        x, y = self.get_dis_xy(volatile=True)
        preds = self.discriminator(x.detach())
        # loss = F.cross_entropy(preds, y)
        loss = F.binary_cross_entropy(preds, y)
        # print(loss)
        stats['DIS_COSTS'].append(loss.detach().item())

        # check NaN
        if (loss != loss).detach().any():
            logger.error("NaN detected (discriminator)")
            sys.exit()

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()
        clip_parameters(self.discriminator, self.params.dis_clip_weights)

    def gen_step(self):
        """
        Fooling discriminator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        self.discriminator.eval()
        self.generator.train()

        # loss
        x, y = self.get_dis_xy(volatile=False)
        preds = self.discriminator(x)
        # loss = self.params.dis_lambda * F.cross_entropy(preds, 1-y)
        loss = self.params.dis_lambda * F.binary_cross_entropy(preds, 1-y)
        # print(loss)

        # check NaN
        if (loss != loss).detach().any():
            logger.error("NaN detected (fool discriminator)")
            sys.exit()

        # optim
        self.gen_optimizer.zero_grad()
        loss.backward()
        self.gen_optimizer.step()
        self.generator.orthogonalize()

        return self.params.langnum * self.params.batch_size

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
        src_emb = self.generator(self.src_emb.weight).detach()
        tgt_emb = self.generator(self.tgt_emb.weight).detach()
        third_emb = self.generator(self.third_emb.weight).detach()
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        third_emb = third_emb / third_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self._src_dico = build_dictionary(src_emb, third_emb, self.params)
        self._tgt_dico = build_dictionary(src_emb, third_emb, self.params)

    def procrustes(self):
        """
        Find the best orthogonal matrix generator using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.src_emb.weight.detach()[self._src_dico[:, 0]]
        B = self.tgt_emb.weight.detach()[self._tgt_dico[:, 0]]
        C1 = self.third_emb.weight.detach()[self._src_dico[:, 1]]
        C2 = self.third_emb.weight.detach()[self._tgt_dico[:, 1]]
        W1 = self.src_generator.weight.detach()
        W2 = self.tgt_generator.weight.detach()
        M1 = C1.transpose(0, 1).mm(A).cpu().numpy()
        M2 = C2.transpose(0, 1).mm(B).cpu().numpy()
        U1, S1, V_t1 = scipy.linalg.svd(M1, full_matrices=True)
        U2, S2, V_t2 = scipy.linalg.svd(M2, full_matrices=True)
        W1.copy_(torch.from_numpy(U1.dot(V_t1)).type_as(W1))
        W2.copy_(torch.from_numpy(U2.dot(V_t2)).type_as(W2))

    # def orthogonalize(self):
    #     """
    #     Orthogonalize the generator.
    #     """
    #     if self.params.gen_beta > 0:
    #         beta = self.params.gen_beta
    #         for i in range(self.params.langnum-1):
    #             W = self.genarator[i].weight.detach()
    #             W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

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
            if to_log[metric] < self.best_valid_metric:
                logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                            , to_log[metric], self.best_valid_metric)
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
                if self.decrease_lr:
                    old_lr = self.gen_optimizer.param_groups[0]['lr']
                    self.gen_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                    logger.info("Shrinking the learning rate: %.5f -> %.5f"
                                , old_lr, self.gen_optimizer.param_groups[0]['lr'])
                self.decrease_lr = True

    # def save_best(self, to_log, metric):
    #     """
    #     Save the best model for the given validation metric.
    #     """
    #     # best generator for the given validation criterion
    #     if to_log[metric] > self.best_valid_metric:
    #         # new best generator
    #         self.best_valid_metric = to_log[metric]
    #         logger.info('* Best value for "%s": %.5f', metric, to_log[metric])
    #         # save the generator

    #         W = self.generator.weight.detach().cpu().numpy()
    #         path = os.path.join(self.params.exp_path, 'best_generator.pth')
    #         logger.info('* Saving the generator to %s ...', path)
    #         torch.save(W, path)

    # def reload_best(self):
    #     """
    #     Reload the best generator.
    #     """
    #     path = os.path.join(self.params.exp_path, 'best_generator.pth')
    #     logger.info('* Reloading the best model from %s ...', path)
    #     # reload the model
    #     assert os.path.isfile(path)
    #     to_reload = torch.from_numpy(torch.load(path))
    #     W = self.generator.weight.detach()
    #     assert to_reload.size() == W.size()
    #     W.copy_(to_reload.type_as(W))

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
    #             embs[j][k:k + bs] = self.genarator[j](x).detach().cpu()

    #     # write embeddings to the disk
    #     export_embeddings(embs, params)
