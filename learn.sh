for smooth in 0.01 0.1 0.2 0.3 0
do
  for step in 1 3 5
  do
    for seed in 0 1 2
    do
      python unsupervised.py --exp_id n_class_-F_smoth${smooth}_step${step}_seed${seed} --seed $seed --dis_steps $step --dis_smooth $smooth --n_epochs 15 --epoch_size 300000
    done
  done
done