#!/bin/bash

# Script to remove function tags from constituent labels
# Probably there's are better way than regex...
for SPLIT in "train" "dev" "test"
do
    IN_DIR=/s0/ttmt001/wsj/$SPLIT-rmtrace
    OUT_DIR=/s0/ttmt001/wsj/$SPLIT-rmfntags
    mkdir -p ${OUT_DIR}
    FILES=`ls ${IN_DIR}/tree2seq_*.txt`
    for f in $FILES
    do 
        BASE=`basename $f`
        OUTFILE=${OUT_DIR}/$BASE
        echo $OUTFILE
        # getting rid of stuff in form AB-CD
        cat $f | sed -E -e "s/(\([A-Z]+)[-=][A-Z0-9]* /\1 /g" > /tmp/tmp.txt
        # getting rid of stuff in form AB-CD-EF 
        cat /tmp/tmp.txt | sed -E -e "s/(\([A-Z]+)[-=][A-Z0-9]*[-=][A-Z0-9]* /\1 /g" > /tmp/tmp-2.txt 
        # and so on, up to 5
        cat /tmp/tmp-2.txt | sed -E -e "s/(\([A-Z]+)[-=][A-Z0-9]*[-=][A-Z0-9]*[-=][A-Z0-9]* /\1 /g" > /tmp/tmp-3.txt 
        cat /tmp/tmp-3.txt | sed -E -e "s/(\([A-Z]+)[-=][A-Z0-9]*[-=][A-Z0-9]*[-=][A-Z0-9]*[-=][A-Z0-9]* /\1 /g" > $OUTFILE 
    done
done

