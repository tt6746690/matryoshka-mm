#!/bin/bash

set -e
set -x

CKPT=$1
TOKEN_SCALE=$2
SAVE_DIR=$3
CONV_MODE=v1
EVAL_DATA_DIR=/fsx/wpq/.data/eval/MME

python -m llava.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file $EVAL_DATA_DIR/llava_mme.jsonl \
    --image-folder $EVAL_DATA_DIR/MME_Benchmark_release_version \
    --answers-file $SAVE_DIR/answers.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE  \
    $(if [ -n "$TOKEN_SCALE" ]; then echo "--matryoshka_vis_token_scale $TOKEN_SCALE"; fi)

python $EVAL_DATA_DIR/convert_answer_to_mme.py \
    --data_path $EVAL_DATA_DIR/MME_Benchmark_release_version \
    --result_dir $SAVE_DIR/

python $EVAL_DATA_DIR/eval_tool/calculation.py --results_dir $SAVE_DIR