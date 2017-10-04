#!/bin/bash

# Script for condensing WSJ trees into one sentence/line format
MAIN_DIR=/g/ssli/data/treebank/release3/parsed/mrg/wsj
OUT_DIR=/s0/ttmt001/wsj/from-treebank
for indir in 00  01  02  03  04  05  06  07  08  09  10  \
    11  12  13  14  15  16  17  18  19  20  21  22  23  24;
do
    python tree2seq.py --indir ${MAIN_DIR}/${indir} \
        --outfile ${OUT_DIR}/tree2seq_${indir}.txt
done

