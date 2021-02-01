"""unsupervised MUSE"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# python unsupervised.py --exp_name en_es_random -exp_id lang_mean_lr0_p.7 --langs en_es_random --emb_init lang_mean --device 2 --emb_lr 0 
# python unsupervised.py --exp_name five_langs --exp_id new_fr --langs de_es_it_pt_fr --device cuda:1
# python unsupervised.py --exp_name six_langs --exp_id en_learnable_lr1_p.5 --langs de_es_it_fr_pt_en --device cuda:1 --emb_lr 1 --n_epochs 15 --dis_sampling 0.5 --eval_type no_target --last_eval no_target --random_start 5
# python unsupervised.py --exp_name seven_langs_new --exp_id en_p.5 --langs ja_de_es_it_fr_pt_en --device cuda:1 --n_epochs 10 --dis_sampling .5 --eval_type all --last_eval all
# python unsupervised.py --exp_name sixes --exp_id six_langs_new2 --langs de_es_it_fr_pt_en --device cuda:2 --eval_type no --last_eval all
# python unsupervised.py --exp_name three_langs_new --exp_id ja_fr_en --langs ja_fr_en --device cuda:2
import os
import time
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator



# main
parser = argparse.ArgumentParser(description='Unsupervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--device", type=str, default='cuda:0', help="select device")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
parser.add_argument("--eval_type", type=str, default="only_target", help="evaluation type during training")
parser.add_argument("--last_eval", type=str, default="all", help="evaluation type last")
parser.add_argument("--test", type=bool, default=False, help="test or not")
# data
parser.add_argument("--langs", type=str, default='es_en', help="Source language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--random_vocab", type=int, default=0, help="Random vocabulary size (0 to disable)")
parser.add_argument("--learnable", type=bool, default=False, help="learn last emb")
# mapping
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
# random embedding
parser.add_argument("--emb_init", type=str, default='uniform', help="initialize type of embeddings")
parser.add_argument("--emb_norm", type=float, default=0, help="norm of embeddings")
# discriminator
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
parser.add_argument("--dis_sampling", type=float, default=0.3, help="probality of learning discriminator")
parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0, help="Discriminator smooth predictions")
parser.add_argument("--clip_grad", type=float, default=1, help="Clip model grads (0 to disable)")
# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--random_start", type=int, default=100, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=1000000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
parser.add_argument("--emb_lr", type=float, default=0, help="rate for learning embeddings")
parser.add_argument("--entropy_coef", type=float, default=1, help="loss entropy term coefficient")
parser.add_argument("--lr_decay", type=float, default=0.95, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement epochs (0 to disable the refinement procedure)")
parser.add_argument("--ref_steps", type=int, default=30000, help="Number of refinement iterations (0 to disable the refinement procedure)")
parser.add_argument("--ref_optimizer", type=str, default="adam", help="refine optimizer")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")


# parse parameters
params = parser.parse_args()

# check parameters
assert params.device == 'cpu' or torch.cuda.is_available()
assert 0 <= params.dis_dropout < 1
assert 0 <= params.dis_input_dropout < 1
assert 0 <= params.dis_smooth < 0.5
assert 0 < params.lr_shrink <= 1
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

params.metric_size = 10000
VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-'+str(params.metric_size)
# VALIDATION_METRIC = 'precision_at_1-csls_knn_10'

# build model / trainer / evaluator
params.langs = params.langs.split('_')
params.langnum = len(params.langs)
params.embpaths = []
for i in range(params.langnum):
    params.embpaths.append('data/wiki.{}.vec'.format(params.langs[i]))
params.emb_optimizer = "sgd,lr=" + str(params.emb_lr)
logger = initialize_exp(params)
mappings, embedding, discriminator = build_model(params)
trainer = Trainer(mappings, embedding, discriminator, params)
evaluator = Evaluator(trainer)


# Learning loop for Adversarial Training
if params.adversarial:
    logger.info('----> ADVERSARIAL TRAINING <----\n\n')

    # training loop
    for n_epoch in range(params.n_epochs):

        logger.info('Starting adversarial training epoch %i...', n_epoch)
        tic = time.time()
        n_words_proc = 0
        stats = {'DIS_COSTS': []}
        stats_str = [('DIS_COSTS', 'Discriminator loss')]

        for n_iter in range(0, params.epoch_size, params.batch_size):

            # discriminator training
            if params.dis_sampling < 1:
                if np.random.rand() <= params.dis_sampling:
                    trainer.dis_step(stats)
            else:
                for i in range(int(params.dis_sampling)):
                    trainer.dis_step(stats)

            # mapping training (discriminator fooling)
            n_words_proc += trainer.gen_step(stats, mode='map')

            # log stats
            if n_iter % 500 == 0:
                stats_log = ['%s: %.4f' % (v, np.mean(stats[k])) for k, v in stats_str if len(stats[k])]
                tgt_norm = torch.mean(torch.norm(embedding.embs[-1].weight, dim=1))
                stats_log.append('Target emb Norm: %.4f' % tgt_norm)
                stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
                stats_log = ' - '.join(stats_log)
                logger.info('%06i - %s', n_iter, stats_log)

                # reset
                tic = time.time()
                n_words_proc = 0
                for k, _ in stats_str:
                    del stats[k][:]

        # embeddings / discriminator evaluation
        to_log = OrderedDict({'n_epoch': n_epoch, 'tgt_norm': tgt_norm.item()})
        evaluator.all_eval(to_log, 'no')
        evaluator.eval_dis(to_log)
        logger.info("__log__:%s", json.dumps(to_log))

        # save best model / end of epoch
        trainer.save_best(to_log, VALIDATION_METRIC)
        logger.info('End of epoch %i.\n\n', n_epoch)

        # update the learning rate (stop if too small)
        trainer.update_lr(to_log, VALIDATION_METRIC)
        # if trainer.map_optimizer.param_groups[0]['lr'] < params.min_lr:
            # logger.info('Learning rate < 1e-6. BREAK.')
            # break

    logger.info('The best metric is %.4f, %d epoch, tgt norm is %.4f', trainer.best_valid_metric, trainer.best_epoch, trainer.best_tgt_norm)

# Learning loop for Procrustes Iterative Refinement
if params.n_refinement:
    # Get the best mapping according to VALIDATION_METRIC
    logger.info('----> ITERATIVE REFINEMENT <----\n\n')
    trainer.reload_best()
    to_log = OrderedDict({'best_epoch': trainer.best_epoch, 'tgt_norm': trainer.best_tgt_norm})
    # evaluator.all_eval(to_log, params.last_eval)
    # evaluator.eval_dis(to_log)

    # training loop
    for n_epoch in range(params.n_refinement):

        logger.info('Starting refinement iteration %i...', n_epoch)

        # build a dictionary from aligned embeddings
        trainer.build_dictionary()

        # optimize MPSR
        tic = time.time()
        n_words_ref = 0
        stats = {'REFINE_COSTS': []}
        for n_iter in range(params.ref_steps):
            # mpsr training step
            n_words_ref += trainer.refine_step(stats)
            # log stats
            if n_iter % 500 == 0:
                stats_str = [('REFINE_COSTS', 'Refine loss')]
                stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                             for k, v in stats_str if len(stats[k]) > 0]
                stats_log.append('%i samples/s' % int(n_words_ref / (time.time() - tic)))
                logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))
                # reset
                tic = time.time()
                n_words_ref = 0
                for k, _ in stats_str:
                    del stats[k][:]

        # embeddings evaluation
        to_log = OrderedDict({'n_epoch': 'refine:'+str(n_epoch), 'tgt_norm':''})
        evaluator.all_eval(to_log, params.eval_type)

        # JSON log / save best model / end of epoch
        logger.info("__log__:%s", json.dumps(to_log))
        trainer.save_best(to_log, VALIDATION_METRIC)
        logger.info('End of refinement iteration %i.\n\n', n_epoch)

to_log = OrderedDict()
trainer.reload_best()
evaluator.all_eval(to_log, params.last_eval)
logger.info("__log__:%s", json.dumps(to_log))
# evaluator.eval_dis(to_log)
logger.info('end of the examination')
# export embeddings
# if params.export:
#     trainer.reload_best()
#     trainer.export()
