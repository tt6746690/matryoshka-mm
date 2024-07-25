#!/bin/bash

set -e
set -x

CKPT=$1
TOKEN_SCALE=$2
SAVE_DIR=$3
CONV_MODE=v1
EVAL_DATA_DIR=/fsx/wpq/.data/eval/mmbench
SPLIT="mmbench_dev_20230712"
ANSWER_UPLOAD_DIR=/fsx/wpq/github/metasummer2024/external/LLaVA/playground/answers_upload

python -m llava.eval.model_vqa_mmbench \
    --model-path $CKPT \
    --question-file $EVAL_DATA_DIR/$SPLIT.tsv \
    --answers-file $SAVE_DIR/$SPLIT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode $CONV_MODE  \
    --matryoshka_vis_token_scale $TOKEN_SCALE

python -m llava.eval.convert_mmbench_for_submission \
    --annotation-file $EVAL_DATA_DIR/$SPLIT.tsv \
    --result-dir $SAVE_DIR \
    --upload-dir $SAVE_DIR \
    --experiment $SPLIT


python -m llava.eval.copy_predictions $CKPT $ANSWER_UPLOAD_DIR