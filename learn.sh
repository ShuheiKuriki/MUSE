for entropy in .0
do
  for sample in .2 .5 1.
  do
    python unsupervised.py --exp_id n_class_entorpy${entropy}_sample${sample} --dis_sampling $sample --n_epochs 15 --epoch_size 500000 --exp_name search --entropy_lambda $entropy
  done
done