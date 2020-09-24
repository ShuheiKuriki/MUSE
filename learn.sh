lgs1="en"
lgs2="de"
for lg1 in ${lgs1}
do
  for lg2 in ${lgs2}
  do
    if [ $lg1 != $lg2 ]; then
      python unsupervised.py --src_lang $lg1 --tgt_lang $lg2
    fi
  done
done