#!/bin/bash
lgs=("en" "ja" "ko" "ru" "hi" "fi")
nums="1 2 3 4 5"
for num in ${nums}
do
  for ((i=0; i<6; i++))
  do
    for ((j=0; j<6; j++))
    do
      if [ $i != $j ]; then
        python map_embeddings.py --unsupervised --cuda ../data/wiki.${lgs[i]}.vec ../data/wiki.${lgs[j]}.vec ../dumped/uncontext/vecmap/${lgs[i]}-${lgs[j]}_${lgs[i]}_${num}.txt ../dumped/uncontext/vecmap/${lgs[i]}-${lgs[j]}_${lgs[j]}_${num}.txt

        python evaluate.py --src_lang ${lgs[i]} --tgt_lang ${lgs[j]} --exp_name uncontext/vecmap --exp_id ${lgs[i]}_${lgs[j]}_$num --num $num --device cuda:0

        rm ../dumped/uncontext/vecmap/${lgs[i]}-${lgs[j]}_${lgs[i]}_${num}.txt
        rm ../dumped/uncontext/vecmap/${lgs[i]}-${lgs[j]}_${lgs[j]}_${num}.txt
      fi
    done
  done
done