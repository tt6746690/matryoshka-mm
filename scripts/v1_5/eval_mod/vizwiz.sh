#!/bin/bash

set -e

CKPT=$1
TOKEN_SCALE=$2
SAVE_DIR=$3
CONV_MODE=v1

EVAL_DATA_DIR=/fsx/wpq/.data/eval/vizwiz

python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file $EVAL_DATA_DIR/llava_test.jsonl \
    --image-folder $EVAL_DATA_DIR/test \
    --answers-file $SAVE_DIR/answers.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE  \
    $(if [ -n "$TOKEN_SCALE" ]; then echo "--matryoshka_vis_token_scale $TOKEN_SCALE"; fi)

python -m llava.eval.convert_vizwiz_for_submission \
    --annotation-file $EVAL_DATA_DIR/llava_test.jsonl \
    --result-file $SAVE_DIR/answers.jsonl \
    --result-upload-file $SAVE_DIR/answers_upload.json


# submit with evalai-cli
source /fsx/wpq/.profile_local.sh
conda activate evalai-cli
echo -e "y\n$CKPT\n\n\n\n" | evalai challenge 2185 phase 4336 submit --file $SAVE_DIR/answers_upload.json  --large --private

