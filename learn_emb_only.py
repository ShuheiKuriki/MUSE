"""unsupervised MUSE"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# python learn_emb_only.py --langs de_pt_random --exp_name learn_map_w_given_emb5/de_pt --exp_id random_vector2 --emb_init norm_mean --emb_norm 4.5 --emb_optimizer adagrad --dis_sampling 1 --n_epochs 10 --device cuda:0
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
# data
parser.add_argument("--langs", type=str, default='es_en', help="Source language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--random_vocab", type=int, default=75000, help="Random vocabulary size (0 to disable)")
parser.add_argument("--learnable", type=bool_flag, default=True, help="whether or not random embedding is learnable")
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
parser.add_argument("--dis_sampling", type=float, default=1, help="probality of learning discriminator")
parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0, help="Discriminator smooth predictions")
parser.add_argument("--clip_grad", type=float, default=1, help="Clip model grads (0 to disable)")
# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=1000000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--emb_optimizer", type=str, default="adam", help="Embedding optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
parser.add_argument("--emb_lr", type=float, default=1., help="rate for learning embeddings")
parser.add_argument("--entropy_coef", type=float, default=1, help="loss entropy term coefficient")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
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

# build model / trainer / evaluator
params.test = False
params.langs = params.langs.split('_')
if params.langs[-1] != 'random':
    params.random_vocab = False
params.langnum = len(params.langs)
params.embpaths = []
for i in range(params.langnum):
    params.embpaths.append('data/wiki.{}.vec'.format(params.langs[i]))
if params.emb_optimizer == 'sgd':
    params.emb_optimizer = "sgd,lr=" + str(params.emb_lr)
logger = initialize_exp(params)
mappings, embedding, discriminator = build_model(params)
trainer = Trainer(mappings, embedding, discriminator, params)
evaluator = Evaluator(trainer)


# Learning loop for Adversarial Training
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
        n_words_proc += trainer.gen_step(stats, mode='emb')

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
    evaluator.all_eval(to_log, '')
    evaluator.eval_dis(to_log)

    # save best model / end of epoch
    trainer.save_best(to_log, VALIDATION_METRIC)
    logger.info('End of epoch %i.\n\n', n_epoch)

    # update the learning rate (stop if too small)
    trainer.update_lr(to_log, VALIDATION_METRIC)

logger.info('The best metric is %.4f, %d epoch, tgt norm is %.4f', trainer.best_valid_metric, trainer.best_epoch, trainer.best_tgt_norm)
path = os.path.join(params.exp_path, 'vectors-%s.pth' % params.langs[-1])
logger.info('Writing source embeddings to %s ...', path)
torch.save(embedding.embs[-1].weight.data.to('cpu'), path)

# to_log = OrderedDict()
# trainer.reload_best()
# evaluator.all_eval(to_log, '')
# evaluator.eval_dis(to_log)
# logger.info('end of the examination')
# export embeddings
# if params.export:
    # trainer.reload_best()
    # trainer.export()
