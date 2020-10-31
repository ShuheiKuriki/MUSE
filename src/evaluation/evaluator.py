"""evaluate embeddings"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from copy import deepcopy
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
        self.dicos = trainer.dicos
        self.generator = trainer.generator
        self.discriminator = trainer.discriminator
        self.params = trainer.params

    def monolingual_wordsim(self, i, to_log):
        """
        Evaluation on monolingual word similarity.
        """
        if i < self.params.langnum-1:
            emb = self.generator(self.embs[i].weight.detach(), i)
        else:
            emb = self.embs[i].weight.detach()
        ws_scores = get_wordsim_scores(
            self.dicos[i].lang, self.dicos[i].word2id, emb.detach().cpu().numpy()
        )
        if ws_scores is not None:
            ws_monolingual_scores = np.mean(list(ws_scores.values()))
            logger.info("Monolingual word similarity score average: %.5f", ws_monolingual_scores)
            to_log['ws_monolingual_scores'] = ws_monolingual_scores
            to_log.update({'src_' + k: ws_scores[k] for k in ws_scores})

    def monolingual_wordanalogy(self, to_log):
        """
        Evaluation on monolingual word analogy.
        """
        src_analogy_scores = get_wordanalogy_scores(
            self.src_dico.lang, self.src_dico.word2id,
            self.src_mapping(self.src_emb.weight).detach().cpu().numpy()
        )
        if self.params.tgt_lang:
            tgt_analogy_scores = get_wordanalogy_scores(
                self.tgt_dico.lang, self.tgt_dico.word2id,
                self.tgt_mapping(self.tgt_emb.weight).detach().cpu().numpy()
            )
        if src_analogy_scores is not None:
            src_analogy_monolingual_scores = np.mean(list(src_analogy_scores.values()))
            logger.info("Monolingual source word analogy score average: %.5f", src_analogy_monolingual_scores)
            to_log['src_analogy_monolingual_scores'] = src_analogy_monolingual_scores
            to_log.update({'src_' + k: v for k, v in src_analogy_scores.items()})
        if self.params.tgt_lang and tgt_analogy_scores is not None:
            tgt_analogy_monolingual_scores = np.mean(list(tgt_analogy_scores.values()))
            logger.info("Monolingual target word analogy score average: %.5f", tgt_analogy_monolingual_scores)
            to_log['tgt_analogy_monolingual_scores'] = tgt_analogy_monolingual_scores
            to_log.update({'tgt_' + k: v for k, v in tgt_analogy_scores.items()})

    def crosslingual_wordsim(self, i, to_log):
        """
        Evaluation on cross-lingual word similarity.
        """
        src_emb = self.generator(self.embs[i].weight.detach(), i).detach().cpu().numpy()
        tgt_emb = self.embs[-1].weight.detach().cpu().numpy()
        # cross-lingual wordsim evaluation
        src_tgt_ws_scores = get_crosslingual_wordsim_scores(
            self.dicos[i].lang, self.dicos[i].word2id, src_emb,
            self.dicos[-1].lang, self.dicos[-1].word2id, tgt_emb
        )
        if src_tgt_ws_scores is None:
            return
        ws_crosslingual_scores = np.mean(list(src_tgt_ws_scores.values()))
        logger.info("Cross-lingual word similarity score average: .%5f", ws_crosslingual_scores)
        to_log['ws_crosslingual_scores'] = ws_crosslingual_scores
        to_log.update({'src_tgt_' + k: v for k, v in src_tgt_ws_scores.items()})

    def word_translation(self, i, to_log):
        """
        Evaluation on word translation.
        """
        # mapped word embeddings
        src_emb = self.generator(self.embs[i].weight.detach(), i).detach()
        tgt_emb = self.embs[-1].weight.detach()

        for method in ['nn', 'csls_knn_10']:
            results = get_word_translation_accuracy(
                self.dicos[i].lang, self.dicos[i].word2id, src_emb, 
                self.dicos[-1].lang, self.dicos[-1].word2id, tgt_emb, 
                method=method, dico_eval=self.params.dico_eval
            )
            to_log.update([('%s-%s' % (k, method), v) for k, v in results])

    def sent_translation(self, i, to_log):
        """
        Evaluation on sentence translation.
        Only available on Europarl, for en - {de, es, fr, it} language pairs.
        """
        lg1 = self.dicos[i].lang
        lg2 = self.dicos[-1].lang

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
        if not self.europarl_data:
            return

        # mapped word embeddings
        src_emb = self.generator(self.embs[i].weight.detach(), i).detach()
        tgt_emb = self.embs[-1].weight.detach()
        # get idf weights
        idf = get_idf(self.europarl_data, lg1, lg2, n_idf=n_idf)

        for method in ['nn', 'csls_knn_10']:

            # source <- target sentence translation
            results = get_sent_translation_accuracy(
                self.europarl_data,
                self.dicos[i].lang, self.dicos[i].word2id, src_emb,
                self.dicos[-1].lang, self.dicos[-1].word2id, tgt_emb,
                n_keys=n_keys, n_queries=n_queries,
                method=method, idf=idf
            )
            to_log.update([('tgt_to_src_%s-%s' % (k, method), v) for k, v in results])

            # target <- source sentence translation
            results = get_sent_translation_accuracy(
                self.europarl_data,
                self.dicos[i].lang, self.dicos[i].word2id, tgt_emb,
                self.dicos[-1].lang, self.dicos[-1].word2id, src_emb,
                n_keys=n_keys, n_queries=n_queries,
                method=method, idf=idf
            )
            to_log.update([('src_to_tgt_%s-%s' % (k, method), v) for k, v in results])

    def dist_mean_cosine(self, i, to_log):
        """
        Mean-cosine model selection criterion.
        """
        # get normalized embeddings
        
        if i == self.params.langnum-1:
            src_emb = self.embs[i].weight.detach()
        else:
            src_emb = self.generator(self.embs[i].weight.detach(), i).detach()
        tgt_emb = self.embs[-1].weight.detach()
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)

        # build dictionary
        for dico_method in ['nn', 'csls_knn_10']:
            dico_build = 'S2T'
            dico_max_size = 10000
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
            logger.info("Mean cosine (%s method, %s build, %i max size): %.5f", dico_method, _params.dico_build, dico_max_size, mean_cosine)
            if i==0:
                to_log['mean_cosine-%s-%s-%i' % (dico_method, _params.dico_build, dico_max_size)] = mean_cosine
            else:
                to_log['mean_cosine-%s-%s-%i' % (dico_method, _params.dico_build, dico_max_size)] += mean_cosine

    def all_eval(self, to_log):
        """
        Run all evaluations.
        """
        self.generator.eval()
        for i in range(self.params.langnum-1):
            logger.info('evaluate %s', self.params.langs[i])
            # self.monolingual_wordsim(i, to_log)
            # for j in range(self.params.langnum):
                # if i == j:
                    # continue
                # logger.info('evaluate %s %s', self.params.langs[i], self.params.langs[j])
            self.crosslingual_wordsim(i, to_log)
            self.word_translation(i, to_log)
            self.sent_translation(i, to_log)
            self.dist_mean_cosine(i, to_log)

    def eval_dis(self, to_log):
        """
        Evaluate discriminator predictions and accuracy.
        """
        bs = 128
        langnum = self.params.langnum
        preds_ = [[] for _ in range(langnum)]
        pred_ = [0]*langnum
        self.discriminator.eval()

        for i in range(langnum):
            for j in range(0, self.embs[i].num_embeddings, bs):
                emb = self.embs[i].weight[j:j + bs].detach()
                if i < langnum-1:
                    preds = self.discriminator(self.generator.mappings[i](emb).detach())
                else:
                    preds = self.discriminator(emb)
                preds_[i].extend(preds.detach().cpu().tolist())
            pred_[i] = np.mean(preds_[i])
            # pred_[i] = np.mean([x[i] for x in preds_[i]])
            # print(preds_[i][0])

            logger.info("Discriminator %s predictions: %.5f", self.params.langs[i], pred_[i])

        accus = [0]*langnum
        cnt = 0
        total = sum([self.embs[i].num_embeddings for i in range(langnum)])
        for i in range(langnum):
            accus[i] = np.mean([x < 0.5 for x in preds_[i]])
            # accus[i] = np.mean([x[i] >= 0.5 for x in preds_[i]])
            cnt += accus[i] * self.embs[i].num_embeddings
            logger.info("Discriminator %s accuracy: %.5f", self.params.langs[i], accus[i])
        dis_accu = cnt/total
        logger.info("Discriminator global accuracy: %.5f", accus[i])

        to_log['dis_accu'] = dis_accu
        # to_log['dis_src_pred'] = pred
        # to_log['dis_tgt_pred'] = pred
