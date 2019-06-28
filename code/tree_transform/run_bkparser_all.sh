#!/bin/bash

DATA_DIR=/s0/ttmt001/speech_parsing/prosodic-anomalies/debug
PARSER_DIR=/homes/ttmt001/transitory/seq2seq_parser/berkeley-parser-analyser/berkeley_parser

FILES=`ls ${DATA_DIR}/*.ms`
for TEST_FILE in $FILES 
do
    FILE_ID=${TEST_FILE##*/}
    FILE_ID=${FILE_ID%.ms}
    # Note: FILE_ID should have the format: FILE_ID="4608_A33_0"
    echo $FILE_ID
    OUT_FILE=${DATA_DIR}/${FILE_ID}.ms.out_with_score
    PRED_FILE=${DATA_DIR}/${FILE_ID}.ms.bkout

    java -client -jar $PARSER_DIR/BerkeleyParser-1.7.jar \
        -gr $PARSER_DIR/swbd.gr \
        -inputFile $TEST_FILE \
        -outputFile $OUT_FILE \
        -kbest 10 -confidence 

    cut -f2 ${OUT_FILE} > ${PRED_FILE}
done

