#!/bin/bash
for ((i=1; i<3; i++))
do
  for ((j=5; j<75; j=j+5))
  do
    python unsupervised.py --exp_id multiply${j}_sampling1_${i} --dis_sampling 1 --exp_name embedding_search --device 2 --random_vocab 0
  done
done