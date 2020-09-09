lgs1="en"
lgs2="it"
for lg1 in ${lgs1}
do
  for lg2 in ${lgs2}
  do
    if [ $lg1 != $lg2 ]; then
      scp -r kuriki@nereus.cl.rcast.u-tokyo.ac.jp:/home/kuriki/MUSE/dumped/learning/$lg1-$lg2-unsup ~/Documents/Github/emojiApp/vectors/$lg1-$lg2-unsup &
    fi
  done
done