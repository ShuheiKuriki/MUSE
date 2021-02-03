"""supervised learning"""
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# python supervised.py --exp_name twos/supervised --exp_id de_es --langs de es --device cuda:0

import os
import json
import argparse
from collections import OrderedDict
import torch

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator

# main
parser = argparse.ArgumentParser(description='Supervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--device", type=str, default='cuda:0', help="select device")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
parser.add_argument("--test", type=bool, default=False, help="test or not")
# data
parser.add_argument("--langs", type=str, nargs='+', default=['es', 'en'], help="languages")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--learnable", type=bool_flag, default=False, help="whether or not random embedding is learnable")
parser.add_argument("--random_vocab", type=int, default=0, help="Random vocabulary size (0 to disable)")
# mapping
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_train", type=str, default="default", help="Path to training dictionary (default: use identical character strings)")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=10000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
parser.add_argument("--metric_size", type=int, default=10000, help="size for csls metric")
# reload pre-trained embeddings
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")


# parse parameters
params = parser.parse_args()

# check parameters
assert params.device == 'cpu' or torch.cuda.is_available()
assert params.dico_train in ["identical_char", "default"] or os.path.isfile(params.dico_train)
assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

# build logger / model / trainer / evaluator
params.langnum = 2
params.embpaths = [f'data/wiki.{params.langs[i]}.vec' for i in range(params.langnum)]
logger = initialize_exp(params)
mapping, embedding, _ = build_model(params, False)
trainer = Trainer(mapping, embedding, None, params)
evaluator = Evaluator(trainer)

# load a training dictionary. if a dictionary path is not provided, use a default
# one ("default") or create one based on identical character strings ("identical_char")
trainer.load_training_dico(params.dico_train)

# define the validation metric
VALIDATION_METRIC_SUP = 'precision_at_1-csls_knn_10'
VALIDATION_METRIC_UNSUP = 'mean_cosine-csls_knn_10-S2T-'+str(params.metric_size)

VALIDATION_METRIC = VALIDATION_METRIC_UNSUP if params.dico_train == 'identical_char' else VALIDATION_METRIC_SUP
logger.info("Validation metric: %s", VALIDATION_METRIC)

# Learning loop for Procrustes Iterative Learning
for n_epoch in range(params.n_refinement + 1):

    logger.info('Starting epoch %i...', n_epoch)

    # build a dictionary from aligned embeddings (unless
    # it is the first iteration and we use the init one)
    if n_epoch > 0: trainer.build_dictionary()

    # apply the Procrustes solution
    trainer.procrustes()

    # embeddings evaluation
    to_log = OrderedDict({'n_epoch': n_epoch, 'tgt_norm': torch.mean(torch.norm(embedding.embs[-1].weight, dim=1)).item()})
    evaluator.all_eval(to_log, 'all')

    # JSON log / save best model / end of epoch
    logger.info("__log__:%s", json.dumps(to_log))
    trainer.save_best(to_log, VALIDATION_METRIC)
    logger.info('End of epoch %i.\n\n', n_epoch)

logger.info('The best metric is %.4f, %d epoch, tgt norm is %.4f', trainer.best_valid_metric, trainer.best_epoch, trainer.best_tgt_norm)

# export embeddings
# if params.export:
    # trainer.reload_best()
    # trainer.export()

logger.info('end of the examination')
