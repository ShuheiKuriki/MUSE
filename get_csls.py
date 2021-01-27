# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# python get_csls.py --langs en_ja_de_es_fr_it_pt_random --exp_name analysis --load_path dumped/seven_w_enlike/new_lr0_adam_p.7_75.2_0.79 --exp_id seven_w_enlike
# python get_csls.py --langs en_ja_random --exp_name analysis --load_path dumped/en_ja_enlike/new_lr0_adam_p.7_45.4_0.82 --exp_id en_ja_enlike_10000
# python get_csls.py --langs en_de_es_fr_it_pt_random --exp_name analysis --load_path dumped/six_w_enlike/new_lr0_adam_p1_77.3_0.80 --exp_id six_w_enlike_check
# python get_csls.py --langs en_ja_de_es_fr_it_pt_random --exp_name analysis --load_path dumped/seven_w_enlike/new_lr0_adam_p.7_75.2_0.79 --exp_id seven_w_enlike_200000
import os
import json
import argparse
from collections import OrderedDict
import torch

from src.utils import bool_flag, initialize_exp, load_embeddings, get_nn_avg_dist
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator

# main
parser = argparse.ArgumentParser(description='Supervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--load_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--device", type=str, default='cuda:0', help="select device")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")

# data
parser.add_argument("--langs", type=str, default='es_en', help="Source language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--n_tgt", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")


# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()

params.langs = params.langs.split('_')
params.langnum = len(params.langs)
params.embpaths = []
for i in range(params.langnum):
    params.embpaths.append('data/wiki.{}.vec'.format(params.langs[i]))

logger = initialize_exp(params)

embs, dicos = [0]*params.langnum, [0]*params.langnum
for i in range(params.langnum-1):
    dicos[i], embs[i] = load_embeddings(params, i)
embs[-1] = torch.load(params.load_path+'/vectors-random.pth')

maps = [0]*(params.langnum-1)
for i in range(params.langnum-1):
    maps[i] = torch.from_numpy(torch.load(params.load_path+'/best_mapping{}.pth'.format(i+1)))
    embs[i] = embs[i].mm(maps[i].t())[:params.n_tgt]

n_src = 128
bs = 64
knn = 10

all_scores = [[] for j in range(params.langnum-1)]
all_ids = [[] for j in range(params.langnum-1)]

for j in range(params.langnum-1):
    # average distances to k nearest neighbors
    emb1, emb2 = embs[-1], embs[j]
    average_dist1 = torch.from_numpy(get_nn_avg_dist(emb2, emb1, knn))
    average_dist2 = torch.from_numpy(get_nn_avg_dist(emb1, emb2, knn))
    average_dist1 = average_dist1.type_as(emb1)
    average_dist2 = average_dist2.type_as(emb2)

    # for every source word
    for i in range(0, n_src, bs):

        # compute target words scores
        scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
        scores.mul_(2)
        scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
        kbest_scores, kbest_ids = scores.topk(10, dim=1, largest=True, sorted=True)

        all_scores[j].append(kbest_scores.cpu())
        all_ids[j].append(kbest_ids.cpu())

    all_scores[j] = torch.cat(all_scores[j], 0)
    all_ids[j] = torch.cat(all_ids[j], 0)

for i in range(n_src):
    logger.info(i)
    for j in range(params.langnum-1):
        logger.info(params.langs[j])
        logger.info(' - '.join([dicos[j].id2word[all_ids[j][i][k].item()] for k in range(10)]))