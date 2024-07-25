#!/bin/bash

set -e
set -x

CKPT=$1
TOKEN_SCALE=$2
SAVE_DIR=$3
CONV_MODE=v1
EVAL_DATA_DIR=/fsx/wpq/.data/eval/scienceqa

python -m llava.eval.model_vqa_science \
    --model-path $1 \
    --question-file $EVAL_DATA_DIR/llava_test_CQM-A.json \
    --image-folder $EVAL_DATA_DIR/images/test \
    --answers-file $SAVE_DIR/answers.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE  \
    $(if [ -n "$TOKEN_SCALE" ]; then echo "--matryoshka_vis_token_scale $TOKEN_SCALE"; fi)

python -m llava.eval.eval_science_qa \
    --base-dir $EVAL_DATA_DIR \
    --result-file $SAVE_DIR/answers.jsonl \
    --output-file $SAVE_DIR/outputs.jsonl \
    --output-result $SAVE_DIR/results.jsonl
