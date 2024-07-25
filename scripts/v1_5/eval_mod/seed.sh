#!/bin/bash

set -e
set -x

CKPT=$1
TOKEN_SCALE=$2
SAVE_DIR=$3
CONV_MODE=v1

EVAL_DATA_DIR=/fsx/wpq/.data/eval/seed_bench

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file $EVAL_DATA_DIR/llava-seed-bench.jsonl \
        --image-folder $EVAL_DATA_DIR \
        --answers-file $SAVE_DIR/answers/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $CONV_MODE  \
        $(if [ -n "$TOKEN_SCALE" ]; then echo "--matryoshka_vis_token_scale $TOKEN_SCALE"; fi) &
done

wait

output_file=$SAVE_DIR/answers/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $SAVE_DIR/answers/${CHUNKS}_${IDX}.json >> "$output_file"
done
 
# Evaluate
python -m llava.eval.convert_seed_for_submission \
    --annotation-file $EVAL_DATA_DIR/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file $SAVE_DIR/answers_upload.jsonl