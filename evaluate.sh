lgs=("pt" "es" "fr" "it" "de")
for ((i=0; i<1; i++))
do
  for ((j=1; j<5; j++))
  do
    python evaluate.py --src_lang ${lgs[i]} --tgt_lang ${lgs[j]}
    python evaluate.py --src_lang ${lgs[j]} --tgt_lang ${lgs[i]}
    python evaluate.py --src_lang ${lgs[i]} --tgt_lang ${lgs[j]} --third_lang True
    python evaluate.py --src_lang ${lgs[j]} --tgt_lang ${lgs[i]} --third_lang True
  done
done