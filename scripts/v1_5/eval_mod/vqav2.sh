#!/bin/bash

set -e


CKPT=$1
TOKEN_SCALE=$2
SAVE_DIR=$3
CONV_MODE=v1

SPLIT="llava_vqav2_mscoco_test-dev2015"
EVAL_DATA_DIR=/fsx/wpq/.data/eval/vqav2

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}


output_file=$SAVE_DIR/$SPLIT/answers/merge.jsonl

if [[ ! -f "$SAVE_DIR/$SPLIT/answers/merge.jsonl" ]]; then

    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
            --model-path $CKPT \
            --question-file $EVAL_DATA_DIR/$SPLIT.jsonl \
            --image-folder $EVAL_DATA_DIR/test2015 \
            --answers-file $SAVE_DIR/$SPLIT/answers/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode $CONV_MODE \
            $(if [ -n "$TOKEN_SCALE" ]; then echo "--matryoshka_vis_token_scale $TOKEN_SCALE"; fi) &
    done

    wait

    echo $output_file

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $SAVE_DIR/$SPLIT/answers/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

fi

python -m llava.eval.convert_vqav2_for_submission --src "$output_file" --dst $SAVE_DIR/$SPLIT/answers_upload.json --test_split $EVAL_DATA_DIR/llava_vqav2_mscoco_test2015.jsonl


# submit with evalai-cli
source /fsx/wpq/.profile_local.sh
conda activate evalai-cli
echo -e "y\n$CKPT\n\n\n\n" | evalai challenge 830 phase 1793 submit --file $SAVE_DIR/$SPLIT/answers_upload.json  --large --private

