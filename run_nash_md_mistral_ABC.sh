#!/bin/bash
# Three Nash-MD runs on Mistral: iter i uses prompt dataset iter ((i-1)%3+1) (same ABC schedule as SPPO).
# Nash-MD samples online; no scripts/generate.sh. Each run loads the previous iter checkpoint as policy.
#
# Requires: pip install "trl>=1.0"

set -e

iter_num=3
RUN_TAG="nashmd-promptABC"
BASE_TOKENIZER="mistralai/Mistral-7B-Instruct-v0.2"

for i in $(seq 1 $iter_num); do
    echo "Running Nash-MD Iter ${i}"
    if [ "$i" -eq 1 ]; then
        MODEL="mistralai/Mistral-7B-Instruct-v0.2"
        TOK_ARG=()
    else
        MODEL="checkpoints/${RUN_TAG}-Mistral-7B-Instruct-NashMD-Iter$((i-1))"
        TOK_ARG=(--tokenizer_name_or_path "$BASE_TOKENIZER")
    fi
    OUTPUT_DIR="checkpoints/${RUN_TAG}-Mistral-7B-Instruct-NashMD-Iter${i}"
    PROMPT_NUM=$(( (i - 1) % 3 + 1 ))
    PROMPT_DATASET="UCLA-AGI/data-mistral-7b-instruct-sppo-iter${PROMPT_NUM}"

    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file recipes/accelerate_configs/deepspeed_zero3_4gpu.yaml \
        --main_process_port 2931 \
        scripts/train_nash_md.py \
        --model_name_or_path "$MODEL" \
        "${TOK_ARG[@]}" \
        --dataset "$PROMPT_DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-7 \
        --beta 0.001 \
        --mixture_coef 0.5 \
        --max_new_tokens 512 \
        --max_length 2048
done
