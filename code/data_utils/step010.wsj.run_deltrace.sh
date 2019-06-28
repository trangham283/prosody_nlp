#!/bin/bash

# Script to remove trace (and its parent label) constituents 
# in WSJ data
for SPLIT in "train" "dev" "test"
do
    IN_DIR=/s0/ttmt001/wsj/$SPLIT-raw
    OUT_DIR=/s0/ttmt001/wsj/$SPLIT-rmtrace
    mkdir -p ${OUT_DIR}
    FILES=`ls ${IN_DIR}/tree2seq_*.txt`
    for f in $FILES
    do 
        BASE=`basename $f`
        OUTFILE=${OUT_DIR}/$BASE
        echo $OUTFILE
        python delete_trace_constituents.py \
            --infile $f \
            --outfile $OUTFILE
    done
done

