#!/bin/bash
# Process smaller splits (1/8) of audio files for better post processing

for NUM in `seq 8` 
do
    echo "Working on split" $NUM
    ./comp_all.sh $NUM
done

