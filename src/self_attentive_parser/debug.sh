#!/bin/bash

python src/main.py train --use-words --model-path-base models/debug --numpy-seed 0 --batch-size 2 --d-model 13 --num-heads 2 --d-kv 4 --d-label-hidden 7 --num-layers 1 --attention-dropout 0 --relu-dropout 0 --residual-dropout 0 --tag-emb-dropout 0 --word-emb-dropout 0 --morpho-emb-dropout 0 --char-lstm-input-dropout 0 --elmo-dropout 0 --no-partitioned > small-debug.log

#python3 src/main.py train --use-words --use-tags --model-path-base models/debug --epochs 10 --numpy-seed 0 
