"""evaluate embedding"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# python evaluate.py --langs es_de --exp_name evaluate2 --exp_id es_de

import os
import sys
import argparse
from collections import OrderedDict

sys.path.append("..")
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
parser.add_argument("--num", type=int, default="1", help="Experiment number")
parser.add_argument("--device", type=str, default='cuda:0', help="select device")
# data
parser.add_argument("--src_lang", type=str, default='en', help="src language")
parser.add_argument("--tgt_lang", type=str, default='es', help="tgt language")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
parser.add_argument("--univ_vocab", type=int, default=0, help="Random vocabulary size (0 to disable)")
parser.add_argument("--emb_lr", type=float, default=0, help="rate for learning embeddings")

# reload pre-trained embeddings
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
parser.add_argument("--learnable", type=bool_flag, default=False, help="whether or not random embedding is learnable")
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
# dictionary creation parameters (for refinement)
parser.add_argument("--last_eval", type=str, default="all", help="evaluation type last (no / only_target / no_target / all)")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_eval_build", type=str, default='S2T', help="S2T,T2S,S2T|T2S,S2T&T2S")
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
params.langnum = 2
params.langs = [params.src_lang, params.tgt_lang]
params.embpaths = [
    f'../dumped/uncontext/vecmap/{params.src_lang}-{params.tgt_lang}_{params.src_lang}_{params.num}.txt',
    f'../dumped/uncontext/vecmap/{params.src_lang}-{params.tgt_lang}_{params.tgt_lang}_{params.num}.txt'
]

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
mapping, embedding, _ = build_model(params, False)
trainer = Trainer(mapping, embedding, None, params)
evaluator = Evaluator(trainer)

# run evaluations
to_log = OrderedDict()

evaluator.all_eval(to_log, 'all')

logger.info('end of the examination')
