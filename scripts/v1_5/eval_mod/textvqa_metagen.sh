#!/bin/bash

# use 10k subset of train file.

set -e
set -x


CKPT=$1
TOKEN_SCALE=$2
SAVE_DIR=$3
CONV_MODE=v1
EVAL_DATA_DIR=/fsx/wpq/.data/eval/textvqa


python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file $EVAL_DATA_DIR/llava_textvqa_train_v051_ocr_10k.jsonl \
    --image-folder $EVAL_DATA_DIR/train_images \
    --answers-file $SAVE_DIR/answers.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE \
    $(if [ -n "$TOKEN_SCALE" ]; then echo "--matryoshka_vis_token_scale $TOKEN_SCALE"; fi)