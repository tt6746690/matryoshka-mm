#!/bin/bash

set -e
set -x

CKPT=$1
TOKEN_SCALE=$2
SAVE_DIR=$3
CONV_MODE=v1

SPLIT="llava_gqa_testdev_balanced"
GQADIR="/fsx/wpq/.data/eval/gqa"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}


if [[ ! -f "$SAVE_DIR/$SPLIT/answers/merge.jsonl" ]]; then

    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $GQADIR/$SPLIT.jsonl \
            --image-folder $GQADIR/images \
            --answers-file $SAVE_DIR/$SPLIT/answers/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode $CONV_MODE  \
            $(if [ -n "$TOKEN_SCALE" ]; then echo "--matryoshka_vis_token_scale $TOKEN_SCALE"; fi) &
    done

    wait

    output_file=$SAVE_DIR/$SPLIT/answers/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"
    
    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $SAVE_DIR/$SPLIT/answers/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

fi


python -m llava.eval.convert_gqa_for_eval --src $output_file --dst $SAVE_DIR/$SPLIT/testdev_balanced_predictions.json

python $GQADIR/eval.py --tier $GQADIR/testdev_balanced --predictions $SAVE_DIR/$SPLIT/testdev_balanced_predictions.json
