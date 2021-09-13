#!/bin/bash
ROOT_PATH=${PWD}
DATA_DIR=$ROOT_PATH/data
TRAIN_FILENAME=$DATA_DIR/jsonl/kp20k/kp20k.train.jsonl
VALID_FILENAME=$DATA_DIR/jsonl/kp20k/kp20k.valid.jsonl
TEST_FILENAME=$DATA_DIR/jsonl/kp20k/kp20k.test.jsonl
VOCAB_PATH=$DATA_DIR/vocab_kp20k.txt
DEST_DIR=$DATA_DIR/kp20k/
EXP_NAME=copyrnn_kp20k_basic

# export CUDA_VISIBLE_DEVICES=1

python3 train.py -exp_name $EXP_NAME \
  -train_filename $TRAIN_FILENAME \
  -valid_filename $VALID_FILENAME -test_filename $TEST_FILENAME \
  -batch_size 128 -max_src_len 1500 -learning_rate 1e-3 \
  -token_field title_and_abstract_tokens -keyphrase_field keyword_tokens \
  -given_keyphrases_field given_keyword_tokens -vocab_path /root/KeyphraseExpansion/data/vocab_kp20k.txt -dest_base_dir /root/KeyphraseExpansion/data/kp20k/ \
  -bidirectional -teacher_forcing -copy_net -shuffle -prefetch \
  -schedule_lr -schedule_step 10000 \
  -train_from /root/KeyphraseExpansion/data/kp20k/copyrnn_kp20k_basic-20210825/copyrnn_kp20k_basic_epoch_2_batch_115000.json
 