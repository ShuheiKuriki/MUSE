#!/bin/bash
lgs="en de it pt es"
# cnt=1
for lg1 in $lgs
do
  for lg2 in $lgs
  do
    if [ $lg1 != $lg2 ]; then
        python unsupervised.py --exp_id ${lg1}-${lg2}-fr --dis_sampling 0.3 --n_epochs 15 --epoch_size 1000000 --entropy_lambda 0.1 --langs ${lg1}_${lg2}_fr --exp_name three_langs --device 1
        # cnt=$[$cnt+1]
      fi
      # echo $cnt
      # if [ $[$cnt%5] -eq 0 ]; then
        # echo "sleep 30m"
        # sleep 30m
      # fi
  done
done