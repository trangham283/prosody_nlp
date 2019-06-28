#!/bin/bash
source /homes/ttmt001/transitory/envs/py3.6-gpu/bin/activate
#source /homes/ttmt001/transitory/envs/py3.6-cpu/bin/activate

DATA_DIR=/g/ssli/data/CTS-English/swbd_align/swbd_trees
FEAT_DIR=/g/ssli/data/CTS-English/swbd_align/swbd_features
RESULT_DIR=/homes/ttmt001/transitory/self-attentive-parser/results

MODELS=(3700 3701 3702 3703 3704)
SEEDS=(1 10 100 1000 10000)

# get the length of the arrays
length=${#MODELS[@]}

# TRAINING LOOP
for ((i=0; i<$length; i++)) 
do
    MODEL_NAME=swbd_${MODELS[$i]}
    SEED=${SEEDS[$i]}
    echo "Training seeding config: " $MODEL_NAME $SEED
    python src/main_sparser.py train --use-elmo --freeze \
        --train-path ${DATA_DIR}/swbd_train2.txt \
        --train-sent-id-path ${DATA_DIR}/train2_sent_ids.txt \
        --dev-path ${DATA_DIR}/swbd_dev.txt \
        --dev-sent-id-path ${DATA_DIR}/dev_sent_ids.txt \
        --feature-path ${FEAT_DIR} \
        --model-path-base models/${MODEL_NAME} \
        --speech-features duration,pause,partition,pitch,fbank \
        --sentence-max-len 200 \
        --d-model 1536 \
        --d-kv 96 \
        --elmo-dropout 0.3 \
        --morpho-emb-dropout 0.3 \
        --num-layers 4 \
        --num-heads 8 \
        --epochs 50 --numpy-seed $SEED >> ${MODEL_NAME}.log
done

# EVALUATION LOOP
for ((i=0; i<$length; i++)) 
do
    MODEL_NAME=swbd_${MODELS[$i]}
    MODEL_PATH=`ls models/${MODEL_NAME}*`
    SEED=${SEEDS[$i]}
    for SPLIT in "test" "dev" 
    do
        PRED_PATH=${RESULT_DIR}/${SPLIT}_${MODELS[$i]}_predicted.txt
        TEST_SENT_ID_PATH=${DATA_DIR}/${SPLIT}_sent_ids.txt
        TEST_NAME=swbd_${SPLIT}.txt
        TEST_PATH=${DATA_DIR}/${TEST_NAME}
        echo ${MODEL_PATH}
        echo ${TEST_PATH}
        echo ${PRED_PATH}
        python src/main_sparser.py test \
            --test-path ${TEST_PATH} \
            --test-sent-id-path ${TEST_SENT_ID_PATH} \
            --output-path ${PRED_PATH} \
            --feature-path ${FEAT_DIR} \
            --test-prefix $SPLIT \
            --model-path-base ${MODEL_PATH} >> ${MODEL_NAME}.log
        echo >> ${MODEL_NAME}.log
    done
done

