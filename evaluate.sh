lgs=("de" "es" "fr" "it" "pt")
# third_lgs=("de" "es" "it" "pt")
for ((i=0; i<5; i++))
# for lg1 in ${lgs}
do
  for ((j=0; j<5; j++))
  do
    if [ $i != $j ]; then
      python evaluate_multi.py --langs ${lgs[i]}_${lgs[j]} --exp_name five_w_enlike2/lr.1_p.4_eval --exp_id ${lgs[i]}_${lgs[j]} --device cuda:3 --langlist ${lgs[i]}_${lgs[j]} --map_path dumped/five_w_enlike2/lr.1_p.4
  # python evaluate.py --src_lang ${lgs[j]} --tgt_lang ${lgs[i]} --third_lang "de"
    fi
  done
done