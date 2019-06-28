#!/bin/bash

# Run bk parser on all sentences in SENT_FILE list

SENT_FILE=human-sents-train.txt

while IFS= read -r line
do
    echo ${line}
    ./run_bkparser.sh ${line}
done < ${SENT_FILE}


