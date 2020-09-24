lgs=("en" "de" "es" "it" "fr" "pt")
third_lgs=("de" "es" "it" "pt")
echo "a"
for third_lg in ${third_lgs}
do
  for ((i=0; i<6; i++))
  # for lg1 in ${lgs}
  do
    if [ $third_lg != ${lgs[i]} ]; then
      for ((j=0; j<6; j++))
      # echo "a"
      # for lg2 in ${lgs}
      do
        echo $third_lg
        if [ $i != $j ] && [ $third_lg != ${lgs[j]} ]; then
          # python evaluate.py --src_lang ${lgs[i]}
          # python evaluate.py --src_lang ${lgs[j]} --tgt_lang ${lgs[i]}
          python evaluate.py --src_lang ${lgs[i]} --tgt_lang ${lgs[j]} --third_lang $third_lg
          # python evaluate.py --src_lang ${lgs[j]} --tgt_lang ${lgs[i]} --third_lang "de"
        fi
      done
    fi
  done
done