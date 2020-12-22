#!/bin/bash
for ((j=3; j<8; j=j+2))
do
  python learn_embedding_only.py --exp_id norm_mean3.9_lr${j}_p.3 --emb_norm 3.9 --emb_lr ${j} --dis_sampling 0.3 --exp_name learn_emb_only --device cuda:3 --langs en_es_random --emb_init norm_mean --n_epochs 50
done