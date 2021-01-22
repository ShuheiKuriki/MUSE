"""evaluate embedding"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# python evaluate_multi.py --langs de_es_fr_it_pt --exp_name five_w_enlike2 --exp_id lr.1_p.4_eval --langlist de_es_fr_it_pt --map_path dumped/five_w_enlike2/lr.1_p.4
# python evaluate_multi.py --langs de_es_random --exp_name learn_map_w_given_by_en_emb2/de_es --exp_id new_lr.3_p1_eval --langlist de_es_fr_it_pt --map_path dumped/learn_map_w_given_by_en_emb2/new_lr.3_p1/de_es

import os
import argparse
from collections import OrderedDict

import torch

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator

# main
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--device", type=str, default='cuda:0', help="select device")
# data
parser.add_argument("--langs", type=str, default='es_en', help="Source language")
parser.add_argument("--langlist", type=str, default='es_en', help="Source language")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
parser.add_argument("--random_vocab", type=int, default=0, help="Random vocabulary size (0 to disable)")
parser.add_argument("--emb_lr", type=float, default=0, help="rate for learning embeddings")
parser.add_argument("--map_path", type=str, default="dumped/three_langs/", help="Experiment name")
# reload pre-trained embeddings
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
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

# parse parameters
params = parser.parse_args()

# check parameters
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)

params.metric_size = 10000
VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-'+str(params.metric_size)

params.test = False
params.langs = params.langs.split('_')+['en']
langlist = params.langlist.split('_')
params.langnum = len(params.langs)
params.embpaths = []
for i in range(params.langnum):
    params.embpaths.append('data/wiki.{}.vec'.format(params.langs[i]))
# lang_list = ['fr', 'it', 'es', 'de', 'pt', 'en']
# build logger / model / trainer / evaluator
logger = initialize_exp(params)
mapping, embedding, _ = build_model(params, False)
trainer = Trainer(mapping, embedding, None, params)
evaluator = Evaluator(trainer)

if params.langs[0] != 'en':
    src_path = os.path.join(params.map_path, 'best_mapping{}.pth'.format(langlist.index(params.langs[0])+1))
    logger.info('* Reloading the model from %s ...', src_path)
    assert os.path.isfile(src_path)
    W = mapping.mappings[0].weight.detach()
    W.copy_(torch.from_numpy(torch.load(src_path)).type_as(W))

if params.langs[1] != 'en':
    tgt_path = os.path.join(params.map_path, 'best_mapping{}.pth'.format(langlist.index(params.langs[1])+1))
    logger.info('* Reloading the model from %s ...', tgt_path)
    assert os.path.isfile(tgt_path)
    W = mapping.mappings[1].weight.detach()
    W.copy_(torch.from_numpy(torch.load(tgt_path)).type_as(W))

# run evaluations
to_log = OrderedDict()

evaluator.all_eval(to_log, 'no_target')

# for n_refine in range(params.n_refinement):
#     trainer.build_dictionary()
#     trainer.procrustes()
#     logger.info('End of refine %i.\n', n_refine)

# to_log = OrderedDict()
# evaluator.all_eval(to_log, 'no_target')

for n_refine in range(params.n_refinement):
    trainer.procrustes2(1)
    logger.info('End of refine %i.\n', n_refine)

to_log = OrderedDict()
evaluator.all_eval(to_log, 'no_target')

logger.info('end of the examination')
