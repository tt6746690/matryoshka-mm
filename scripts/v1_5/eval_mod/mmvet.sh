#!/bin/bash

set -e
set -x

CKPT=$1
TOKEN_SCALE=$2
SAVE_DIR=$3
CONV_MODE=v1
EVAL_DATA_DIR=/fsx/wpq/.data/eval/mm-vet

python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file $EVAL_DATA_DIR/llava-mm-vet.jsonl \
    --image-folder $EVAL_DATA_DIR/images \
    --answers-file $SAVE_DIR/answers.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE  \
    $(if [ -n "$TOKEN_SCALE" ]; then echo "--matryoshka_vis_token_scale $TOKEN_SCALE"; fi)

python -m llava.eval.convert_mmvet_for_eval \
    --src $SAVE_DIR/answers.jsonl \
    --dst $SAVE_DIR/results.json

python $EVAL_DATA_DIR/mm-vet_evaluator.py \
    --mmvet_path $EVAL_DATA_DIR \
    --result_file $SAVE_DIR/results.json \
    --result_path $SAVE_DIR \
    --gpt_model gpt-4-0613
    # --use_sub_set to debug