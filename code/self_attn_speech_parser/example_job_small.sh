#!/bin/bash
# Load environment with dependencies
source /homes/ttmt001/transitory/envs/py3.6-cpu/bin/activate

# Data paths
DATA_DIR=sample_data
FEAT_DIR=sample_data
MODEL_DIR=sample_data
RESULT_DIR=sample_data

# TRAINING EXAMPLE
MODEL_NAME=sample
SEED=0
echo "Training seeding config: " $MODEL_NAME $SEED
python src/main_sparser.py train --use-bert --freeze \
    --train-path ${DATA_DIR}/sample_train.txt \
    --train-sent-id-path ${DATA_DIR}/sample_train_sent_ids.txt \
    --dev-path ${DATA_DIR}/sample_dev.txt \
    --dev-sent-id-path ${DATA_DIR}/sample_dev_sent_ids.txt \
    --prefix "sample_" \
    --feature-path ${FEAT_DIR} \
    --model-path-base ${MODEL_DIR}/${MODEL_NAME} \
    --speech-features duration,pause,partition,pitch,fbank \
    --sentence-max-len 200 \
    --d-model 144 \
    --d-kv 24 \
    --elmo-dropout 0.3 \
    --morpho-emb-dropout 0.3 \
    --num-layers 2 \
    --num-heads 4 \
    --epochs 2 --numpy-seed $SEED >> ${MODEL_NAME}.log

# EVALUATION EXAMPLE
MODEL_NAME=sample
MODEL_PATH=${MODEL_DIR}/${MODEL_NAME}

SPLIT="sample_test"
PRED_PATH=${RESULT_DIR}/sample_test_${MODEL_NAME}_predicted.txt
TEST_SENT_ID_PATH=${DATA_DIR}/sample_test_sent_ids.txt
TEST_PATH=${DATA_DIR}/sample_test.txt
python src/main_sparser.py test \
    --test-path ${TEST_PATH} \
    --test-sent-id-path ${TEST_SENT_ID_PATH} \
    --output-path ${PRED_PATH} \
    --feature-path ${FEAT_DIR} \
    --test-prefix $SPLIT \
    --model-path-base ${MODEL_PATH} 

