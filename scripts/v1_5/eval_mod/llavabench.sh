#!/bin/bash

set -e
set -x

CKPT=$1
TOKEN_SCALE=$2
SAVE_DIR=$3
CONV_MODE=v1

EVAL_DATA_DIR=/fsx/wpq/.data/eval/llava-bench-in-the-wild
LLAVA_REPO_DIR=/fsx/wpq/github/metasummer2024/external/LLaVA

python -m llava.eval.model_vqa \
    --model-path $CKPT \
    --question-file $EVAL_DATA_DIR/questions.jsonl \
    --image-folder $EVAL_DATA_DIR/images \
    --answers-file $SAVE_DIR/answers.jsonl \
    --temperature 0 \
    --conv-mode $CONV_MODE  \
    $(if [ -n "$TOKEN_SCALE" ]; then echo "--matryoshka_vis_token_scale $TOKEN_SCALE"; fi)

mkdir -p $EVAL_DATA_DIR/reviews

python -m llava.eval.eval_gpt_review_bench \
    --question $EVAL_DATA_DIR/questions.jsonl \
    --context $EVAL_DATA_DIR/context.jsonl \
    --rule $LLAVA_REPO_DIR/llava/eval/table/rule.json \
    --answer-list \
        $EVAL_DATA_DIR/answers_gpt4.jsonl \
        $SAVE_DIR/answers.jsonl \
    --output \
        $SAVE_DIR/reviews.jsonl

python -m llava.eval.summarize_gpt_review -f $SAVE_DIR/reviews.jsonl