#!/bin/bash
set -e

mkdir ./crosslingual/aligned_europarl
TOOLS=tools
mkdir -p $TOOLS

cd $TOOLS

# fast_align
git clone https://github.com/clab/fast_align
cd fast_align
mkdir build
cd build
cmake ..
make
cd ../../..
set -e

# L1, L2のリストを実験したい言語に合わせて編集
for L1 in en
do
    for L2 in bg el de es fr
    do
        MODEL=bert-base-multilingual-cased
        FAST_ALIGN=./tools/fast_align/build
        OUTPUT_PATH=./crosslingual/europarl
        OUTPUT_PATH2=./crosslingual/aligned_europarl/$L2-$L1
        mkdir $OUTPUT_PATH2
        CACHE=./cache
        PARA=europarl-v7.$L1-$L2.30k

        # download and prep parallel data
        wget https://www.statmt.org/europarl/v7/$L2-$L1.tgz
        tar -zxvf $L2-$L1.tgz -C $OUTPUT_PATH/
        rm -f $L2-$L1.tgz
        python prep_parallel.py --bert_model $MODEL --lang1 $OUTPUT_PATH/europarl-v7.$L2-$L1.$L1 --lang2 $OUTPUT_PATH/europarl-v7.$L2-$L1.$L2 --size 30000 --output $OUTPUT_PATH2/$PARA

        # get word alignment
        $FAST_ALIGN/fast_align -i $OUTPUT_PATH2/$PARA.uncased -d -o -v > $OUTPUT_PATH2/forward.align.$PARA.uncased
        $FAST_ALIGN/fast_align -i $OUTPUT_PATH2/$PARA.uncased -d -o -v -r > $OUTPUT_PATH2/reverse.align.$PARA.uncased
        $FAST_ALIGN/atools -i $OUTPUT_PATH2/forward.align.$PARA.uncased -j $OUTPUT_PATH2/reverse.align.$PARA.uncased -c grow-diag-final-and > $OUTPUT_PATH2/sym.align.$PARA.uncased
    done
done
