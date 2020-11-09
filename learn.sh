lgs="en es fr de"
for lg1 in $lgs
do
  for lg2 in $lgs
  do
    for lg3 in $lgs
    do
      if [ $lg1 != $lg2 ] && [ $lg1 != $lg3 ] && [ $lg2 != $lg3 ]; then
        python unsupervised.py --exp_id ${lg1}-${lg2}-${lg3} --dis_sampling 0.3 --n_epochs 15 --epoch_size 1000000 --entropy_lambda 0.1 --langs ${lg1}_${lg2}_${lg3} --exp_name three_langs --device 3 &
      fi
    done
  done
done