#!/bin/bash
lrs=(1 2 5 10)
for ((j=0; j<4; j++))
do
  python unsupervised.py --exp_id de_pt_en_emblr${lrs[j]}_sampling1 --dis_sampling 1 --emb_lr ${lrs[j]} --exp_name random_search6 --device cuda:2 --langs de_pt_en --learnable True
done