"""evaluate embeddings"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from copy import deepcopy
import json
import numpy as np
import torch
# from torch.autograd import Variable
from torch import Tensor as torch_tensor
import torch.nn.functional as F

from . import get_wordsim_scores, get_crosslingual_wordsim_scores, get_wordanalogy_scores
from . import get_word_translation_accuracy
from . import load_europarl_data, get_sent_translation_accuracy
from ..dico_builder import get_candidates, build_dictionary
from src.utils import get_idf


logger = getLogger()


class Evaluator:
    """Evaluater"""

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.embs = trainer.embs
        self.embedding = trainer.embedding
        self.vocabs = trainer.vocabs
        self.mapping = trainer.mapping
        self.discriminator = trainer.discriminator
        self.params = trainer.params
        self.langnum = self.params.langnum
        # self.num_pairs = (self.langnum-1)*(self.langnum-2)x

    def monolingual_wordsim(self, i):
        """
        Evaluation on monolingual word similarity.
        """
        emb = self.embs[i].weight.detach()
        ws_scores = get_wordsim_scores(self.vocabs[i].lang, self.vocabs[i].word2id, emb.detach().cpu().numpy())
        if ws_scores is not None:
            ws_monolingual_scores = np.mean(list(ws_scores.values()))
            logger.info("Monolingual word similarity score average: %.5f", ws_monolingual_scores)
            # to_log['ws_monolingual_scores'] = ws_monolingual_scores
            # to_log.update({'src_' + k: ws_scores[k] for k in ws_scores})

    def monolingual_wordanalogy(self, i, to_log):
        """
        Evaluation on monolingual word analogy.
        """
        analogy_scores = get_wordanalogy_scores(
            self.vocabs[i].lang, self.vocabs[i].word2id,
            self.mapping(self.embs[i].weight, i).detach().cpu().numpy()
        )
        if analogy_scores is not None:
            analogy_monolingual_scores = np.mean(list(analogy_scores.values()))
            logger.info("Monolingual word analogy score average: %.5f", analogy_monolingual_scores)
            to_log['analogy_monolingual_scores'] = analogy_monolingual_scores
            to_log.update(analogy_scores)

    def crosslingual_wordsim(self, i, j, to_log):
        """
        Evaluation on cross-lingual word similarity.
        """
        src_emb = self.mapping(self.embs[i].weight.detach(), i)
        tgt_emb = self.mapping(self.embs[j].weight.detach(), j)
        src_emb = src_emb.detach().cpu().numpy()
        tgt_emb = tgt_emb.detach().cpu().numpy()
        # cross-lingual wordsim evaluation
        src_tgt_ws_scores = get_crosslingual_wordsim_scores(
            self.vocabs[i].lang, self.vocabs[i].word2id, src_emb,
            self.vocabs[j].lang, self.vocabs[j].word2id, tgt_emb
        )
        if src_tgt_ws_scores is None: return
        ws_crosslingual_scores = np.mean(list(src_tgt_ws_scores.values()))
        logger.info("%s-%s Cross-lingual word similarity score average: .%5f", self.params.langs[i], self.params.langs[j], ws_crosslingual_scores)
        # to_log['ws_crosslingual_scores'] = ws_crosslingual_scores
        for k, v in src_tgt_ws_scores.items():
            if 'src_tgt_' + k in to_log:
                to_log['src_tgt_' + k].append(v)
            else:
                to_log['src_tgt_' + k] = [v]

    def word_translation(self, i, j, to_log):
        """
        Evaluation on word translation.
        """
        # mapped word embeddings
        src_emb = self.mapping(self.embs[i].weight.detach(), i).detach()
        tgt_emb = self.mapping(self.embs[j].weight.detach(), j).detach()
        for method in ['nn', 'csls_knn_10']:
            results = get_word_translation_accuracy(self.vocabs[i].lang, self.vocabs[i].word2id, src_emb, self.vocabs[j].lang, self.vocabs[j].word2id, tgt_emb, method=method, dico_eval=self.params.dico_eval)
            if results is None: return
            # to_log.update([('%s-%s' % (k, method), v) for k, v in results])
            for k, v in results:
                if '{}-{}'.format(k, method) in to_log:
                    to_log['{}-{}'.format(k, method)].append(v)
                else:
                    to_log['{}-{}'.format(k, method)] = [v]

    def sent_translation(self, i, j, to_log):
        """
        Evaluation on sentence translation.
        Only available on Europarl, for en - {de, es, fr, it} language pairs.
        """
        lg1 = self.vocabs[i].lang
        lg2 = self.vocabs[j].lang

        # parameters
        n_keys = 200000
        n_queries = 2000
        n_idf = 300000

        # load europarl data
        if not hasattr(self, 'europarl_data'):
            self.europarl_data = load_europarl_data(
                lg1, lg2, n_max=(n_keys + 2 * n_idf)
            )

        # if no Europarl data for this language pair
        if not self.europarl_data: return

        # mapped word embeddings
        src_emb = self.mapping(self.embs[i].weight.detach(), i).detach()
        tgt_emb = self.mapping(self.embs[j].weight.detach(), j).detach()
        # get idf weights
        idf = get_idf(self.europarl_data, lg1, lg2, n_idf=n_idf)

        for method in ['nn', 'csls_knn_10']:

            # source <- target sentence translation
            results = get_sent_translation_accuracy(
                self.europarl_data,
                self.vocabs[i].lang, self.vocabs[i].word2id, src_emb,
                self.vocabs[j].lang, self.vocabs[j].word2id, tgt_emb,
                n_keys=n_keys, n_queries=n_queries,
                method=method, idf=idf
            )
            to_log.update([('tgt_to_src_%s-%s' % (k, method), v) for k, v in results])

            # target <- source sentence translation
            results = get_sent_translation_accuracy(
                self.europarl_data,
                self.vocabs[i].lang, self.vocabs[i].word2id, tgt_emb,
                self.vocabs[j].lang, self.vocabs[j].word2id, src_emb,
                n_keys=n_keys, n_queries=n_queries,
                method=method, idf=idf
            )
            to_log.update([('src_to_tgt_%s-%s' % (k, method), v) for k, v in results])

    def dist_mean_cosine(self, to_log, i, j):
        """
        Mean-cosine model selection criterion.
        """
        # get normalized embeddings
        src_emb = self.mapping(self.embs[i].weight.detach(), i).detach()
        tgt_emb = self.mapping(self.embs[j].weight.detach(), j).detach()
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

        # build dictionary
        for dico_method in ['csls_knn_10']:
            dico_build = 'S2T'
            dico_max_size = self.params.metric_size
            # temp params / dictionary generation
            _params = deepcopy(self.params)
            _params.dico_method = dico_method
            _params.dico_build = dico_build
            _params.dico_threshold = 0
            _params.dico_max_rank = 10000
            _params.dico_min_size = 0
            _params.dico_max_size = dico_max_size
            s2t_candidates = get_candidates(src_emb, tgt_emb, _params)
            t2s_candidates = get_candidates(tgt_emb, src_emb, _params)
            dico = build_dictionary(src_emb, tgt_emb, _params, s2t_candidates, t2s_candidates)
            # mean cosine
            if dico is None:
                mean_cosine = -1e9
            else:
                mean_cosine = (src_emb[dico[:dico_max_size, 0]] * tgt_emb[dico[:dico_max_size, 1]]).sum(1).mean()
            mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch_tensor) else mean_cosine
            logger.info("%s-%s Mean cosine (%s method, %s build, %i max size): %.5f", self.params.langs[i], self.params.langs[j], dico_method, _params.dico_build, dico_max_size, mean_cosine)
            if 'mean_cosine-{}-{}-{}'.format(dico_method, _params.dico_build, dico_max_size) in to_log:
                to_log['mean_cosine-%s-%s-%i' % (dico_method, _params.dico_build, dico_max_size)].append(mean_cosine)
            else:
                to_log['mean_cosine-%s-%s-%i' % (dico_method, _params.dico_build, dico_max_size)] = [mean_cosine]

    def all_eval(self, to_log, eval_type='no_target'):
        """
        Run all evaluations.
        """
        self.mapping.eval()
        for i in range(self.langnum):
            logger.info('evaluate %s', self.params.langs[i])
            if eval_type == 'no_target':
                if i == self.langnum-1: continue
                for j in range(self.langnum-1):
                    if i == j: continue
                    logger.info('evaluate %s %s', self.params.langs[i], self.params.langs[j])
                    self.crosslingual_wordsim(i, j, to_log)
                    self.word_translation(i, j, to_log)
                    # self.sent_translation(i, j, to_log)
            elif eval_type == 'all':
                self.monolingual_wordsim(i)
                for j in range(self.langnum):
                    if i == j: continue
                    logger.info('evaluate %s %s', self.params.langs[i], self.params.langs[j])
                    self.crosslingual_wordsim(i, j, to_log)
                    self.word_translation(i, j, to_log)
                    # self.sent_translation(i, j, to_log)
            elif eval_type == 'only_target':
                if i == self.langnum-1: continue
                self.crosslingual_wordsim(i, self.langnum-1, to_log)
                self.word_translation(i, self.langnum-1, to_log)
                # self.sent_translation(i, self.langnum-1, to_log)
        for i in range(self.langnum-1):
            for j in range(i+1, self.langnum): self.dist_mean_cosine(to_log, i, j)
        for k in to_log:
            if isinstance(to_log[k], list): to_log[k] = sum(to_log[k])/len(to_log[k])

    def eval_dis(self, to_log):
        """
        Evaluate discriminator predictions and accuracy.
        """
        bs = 128
        langnum = self.langnum
        preds_ = [[] for _ in range(langnum)]
        pred_ = [0]*langnum
        confusion_matrix = [[0]*langnum for _ in range(langnum)]
        self.discriminator.eval()

        for i in range(langnum):
            for j in range(0, self.embs[i].num_embeddings, bs):
                emb = self.embs[i].weight[j:j + bs].detach()
                if i < langnum-1:
                    preds = self.discriminator(self.mapping.linear[i](emb).detach())
                else:
                    preds = self.discriminator(emb)
                preds_[i].extend(torch.exp(preds).detach().cpu().tolist())
            # pred_[i] = np.mean(preds_[i])
            pred_[i] = np.mean([x[i] for x in preds_[i]])
            for x in preds_[i]:
                max_prob = max(x)
                for l in range(langnum):
                    if x[l] == max_prob:
                        confusion_matrix[i][l] += 1

            logger.info("Discriminator %s predictions: %.5f", self.params.langs[i], pred_[i])

        for i in range(langnum):
            for j in range(langnum):
                logger.info("Discriminator true %s pred %s: %.5f", self.params.langs[i], self.params.langs[j], confusion_matrix[i][j]/sum(confusion_matrix[i]))
        dis_accu = sum(a[i] for i, a in enumerate(confusion_matrix))/sum(sum(a) for a in confusion_matrix)

        logger.info("Discriminator global accuracy: %.5f", dis_accu)
        to_log['dis_accu'] = dis_accu
        # to_log['dis_src_pred'] = pred
        # to_log['dis_tgt_pred'] = pred
