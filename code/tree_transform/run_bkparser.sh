#!/bin/bash

DATA_DIR=/s0/ttmt001/speech_parsing/prosodic-anomalies
#DATA_DIR=/homes/ttmt001/transitory/prosodic-anomalies/samples
#FILE_ID="4608_A"
FILE_ID=$1
TEST_FILE=${DATA_DIR}/${FILE_ID}.ms
OUT_FILE=${DATA_DIR}/${FILE_ID}.ms.out_with_score
PRED_FILE=${DATA_DIR}/${FILE_ID}.ms.bkout
PARSER_DIR=/homes/ttmt001/transitory/seq2seq_parser/berkeley-parser-analyser/berkeley_parser

java -client -jar $PARSER_DIR/BerkeleyParser-1.7.jar \
    -gr $PARSER_DIR/swbd.gr \
    -inputFile $TEST_FILE \
    -outputFile $OUT_FILE \
    -kbest 10 -confidence 

cut -f2 ${OUT_FILE} > ${PRED_FILE}
