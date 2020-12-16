#!/bin/bash
for ((i=4; i<8; i=i+3))
do
  # for ((j=5; j<75; j=j+5))
  # do
  python unsupervised.py --exp_id en_es_random_uniform_fixed_sampling${i} --dis_sampling ${i} --exp_name random_search6 --device cuda:1 --langs en_es_random --multiply 8
  # done
done