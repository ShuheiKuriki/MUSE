"""
test code
"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# python test.py --seed 0 --cuda False
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

VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'

# main
parser = argparse.ArgumentParser(description='Unsupervised training')
parser.add_argument("--seed", type=int, default=0, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--test_epochs", type=int, default=5, help="the number of test epochs")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="test", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--device", type=str, default='cuda:0', help="select device cpu or cuda:0,1,2,3")
parser.add_argument("--test", type=bool, default=True, help="test or not")
# data
parser.add_argument("--langs", type=str, nargs='+', default=['es', 'en'], help="languages")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--univ_vocab", type=int, default=0, help="Random vocabulary size (0 to disable)")
parser.add_argument("--random_norm", type=float, default=1., help="multiply random embeddings")
parser.add_argument("--random_init", type=str, default="uniform", help="type of initialize random vectors")
parser.add_argument("--learnable", type=bool_flag, default=False, help="whether or not random embedding is learnable")
# mapping
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
# discriminator
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--clip_grad", type=float, default=1, help="Clip model grads (0 to disable)")
# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--emb_optimizer", type=str, default="sgd", help="Embedding optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer")
parser.add_argument("--emb_lr", type=float, default=1., help="rate for learning embeddings")
parser.add_argument("--entropy_coef", type=float, default=1, help="loss entropy term coefficient")
parser.add_argument("--lr_decay", type=float, default=0.95, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
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
assert params.dis_lambda > 0 and params.dis_steps > 0
assert 0 < params.lr_shrink <= 1
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)

# build model / trainer / evaluator
if params.langs[-1] == 'random': params.univ_vocab = 75000
params.langnum = len(params.langs)
params.embpaths = [f'data/wiki.{params.langs[i]}.vec' for i in range(params.langnum)]
if params.emb_optimizer == 'sgd': params.emb_optimizer = "sgd,lr=" + str(params.emb_lr)
logger = initialize_exp(params)
mapping, embedding, discriminator = build_model(params)
trainer = Trainer(mapping, embedding, discriminator, params)
evaluator = Evaluator(trainer)

# Learning loop for Adversarial Training
logger.info('----> ADVERSARIAL TRAINING <----\n\n')

stats = {'DIS_COSTS': []}
stats_str = [('DIS_COSTS', 'Discriminator loss')]
# discriminator training
for _ in range(params.test_epochs):
    trainer.dis_step(stats)
    trainer.gen_step(mode='map')
    trainer.gen_step(mode='emb')

stats_log = ['%s: %.4f' % (v, np.mean(stats[k])) for k, v in stats_str if len(stats[k])]
tgt_norm = torch.mean(torch.norm(embedding.embs[-1].weight, dim=1))
stats_log.append('Target emb Norm: %.4f' % tgt_norm)
stats_log = ' - '.join(stats_log)
logger.info(stats_log)
