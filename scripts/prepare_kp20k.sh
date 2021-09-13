#!/bin/bash
ROOT_PATH=${PWD}
DATA_DIR=$ROOT_PATH/data
SRC_TRAIN=$DATA_DIR/raw/kp20k/kp20k_training.json
SRC_VALID=$DATA_DIR/raw/kp20k/kp20k_validation.json
SRC_TEST=$DATA_DIR/raw/kp20k/kp20k_testing.json
DEST_TRAIN=$DATA_DIR/jsonl/kp20k/kp20k.train.jsonl
DEST_VALID=$DATA_DIR/jsonl/kp20k/kp20k.valid.jsonl
DEST_TEST=$DATA_DIR/jsonl/kp20k/kp20k.test.jsonl
DEST_VOCAB=$DATA_DIR/vocab_kp20k.txt

python3 preprocess.py -src_filename $SRC_TRAIN \
        -dest_filename $DEST_TRAIN -dest_vocab_path $DEST_VOCAB -src_lower -target_lower

python3 preprocess.py -src_filename $SRC_VALID \
        -dest_filename $DEST_VALID -src_lower -target_lower

python3 preprocess.py -src_filename $SRC_TEST \
        -dest_filename $DEST_TEST -src_lower -target_lower
