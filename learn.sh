for entropy in .2 .3
do
  for sample in .3
  do
    python unsupervised.py --exp_id entropy${entropy}_sample${sample} --dis_sampling $sample --n_epochs 15 --epoch_size 1000000 --exp_name search --entropy_lambda $entropy --langs es_it_en --exp_name es_it_en --device 3
  done
done