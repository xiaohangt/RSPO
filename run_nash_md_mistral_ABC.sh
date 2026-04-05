#!/bin/bash
# Iterative Nash-MD (TRL v1.0+ experimental) on Mistral-7B-Instruct with the same ABC prompt
# cycling and data generation path as run_sppo_mistral_ABC_reg.sh.
#
# Requires: pip install "trl>=1.0" (and project deps for scripts/generate.sh: vllm, PairRM ranking, etc.)

set -e

iter_num=3
RUN_TAG="nashmd-promptABC"

for i in $(seq 1 $iter_num); do
    echo "Running Nash-MD Iter ${i}"
    if [ "$i" -eq 1 ]; then
        MODEL="mistralai/Mistral-7B-Instruct-v0.2"
    else
        MODEL="checkpoints/${RUN_TAG}-Mistral-7B-Instruct-NashMD-Iter$((i-1))"
    fi
    OUTPUT_DIR="checkpoints/${RUN_TAG}-Mistral-7B-Instruct-NashMD-Iter${i}"
    PROMPT_NUM=$(( (i - 1) % 3 + 1 ))
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter${PROMPT_NUM}"
    OUT="data-${RUN_TAG}-mistral-7b-instruct-sppo-iter${i}"
    DATASET_DIR="synthetic_data_${RUN_TAG}-mistral-7b-instruct-sppo-iter${i}_score"

    bash scripts/generate.sh --model "$MODEL" --prompt "$PROMPT" --out_path "$OUT"

    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file recipes/accelerate_configs/deepspeed_zero3_4gpu.yaml \
        --main_process_port 2931 \
        scripts/train_nash_md.py \
        --model_name_or_path "$MODEL" \
        --dataset "$DATASET_DIR" \
        --hub_dataset_org UCLA-AGI \
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
