#!/bin/bash
ROOT_PATH=${PWD}
DATA_DIR=$ROOT_PATH/data


SRC_TEST2=$DATA_DIR/raw/inspec/inspec_test.json
DEST_TEST2=$DATA_DIR/jsonl/inspec/inspec.test50.jsonl

SRC_TEST3=$DATA_DIR/raw/krapivin/krapivin_test.json
DEST_TEST3=$DATA_DIR/jsonl/krapivin/krapivin.test50.jsonl

SRC_TEST4=$DATA_DIR/raw/nus/nus_test.json
DEST_TEST4=$DATA_DIR/jsonl/nus/nus.test50.jsonl

SRC_TEST5=$DATA_DIR/raw/semeval/semeval_test.json
DEST_TEST5=$DATA_DIR/jsonl/semeval/semeval.test50.jsonl

SRC_TEST6=$DATA_DIR/raw/kp20k/kp20k_testing.json
DEST_TEST6=$DATA_DIR/jsonl/kp20k/kp20k.test50.jsonl

DEST_VOCAB=$DATA_DIR/vocab_kp20k.txt


        
python3 preprocess.py -src_filename $SRC_TEST2 \
        -dest_filename $DEST_TEST2 -src_lower -target_lower

python3 preprocess.py -src_filename $SRC_TEST3 \
        -dest_filename $DEST_TEST3 -src_lower -target_lower
        
python3 preprocess.py -src_filename $SRC_TEST4 \
        -dest_filename $DEST_TEST4 -src_lower -target_lower
        
python3 preprocess.py -src_filename $SRC_TEST5 \
        -dest_filename $DEST_TEST5 -src_lower -target_lower
        
python3 preprocess.py -src_filename $SRC_TEST6 \
        -dest_filename $DEST_TEST6 -src_lower -target_lower
