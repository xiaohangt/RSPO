#!/bin/bash
# Three Nash-MD runs on Mistral: iter i uses prompt dataset iter ((i-1)%3+1) (same ABC schedule as SPPO).
# Nash-MD samples online; no scripts/generate.sh. Each run loads the previous iter checkpoint as policy.
#
# Requires: pip install "trl>=1.0" llm-blender
#
# Local PairRM judge (default — no Hugging Face Inference / router.huggingface.co):
#   export JUDGE=pairrm   # default; passes --judge_backend pairrm to train_nash_md.py
# PairRM weights load via llm-blender from the HF cache on first use. To prefetch offline:
#   huggingface-cli download llm-blender/PairRM
# Multi-GPU + PairRM is fragile; for the least friction use: NUM_GPUS=1
# To use the remote Inference API instead (needs working HTTPS + HF token):
#   export JUDGE=hf
# Or stay on pairrm but allow auto-switch when HF_INFERENCE_JUDGE=1 (see train_nash_md.py).

set -e

# Number of GPUs / accelerate processes.
# Default NUM_GPUS=1 so local PairRM works (multi-GPU + PairRM exits unless ALLOW_PAIRRM_MULTIGPU or HF_INFERENCE_JUDGE=1).
# Multi-GPU policy training: NUM_GPUS=4 bash run_nash_md_mistral_ABC.sh
#   and either HF_INFERENCE_JUDGE=1 + HF_TOKEN, or JUDGE=hf, or ALLOW_PAIRRM_MULTIGPU=1 (fragile).
NUM_GPUS="${NUM_GPUS:-1}"
JUDGE="${JUDGE:-pairrm}"
# NUM_GPUS=1: bf16 without DeepSpeed (ZeRO-3 meta init breaks llm-blender PairRM). NUM_GPUS>1: DeepSpeed ZeRO-3.
if [ "${NUM_GPUS}" -eq 1 ]; then
    ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-recipes/accelerate_configs/single_gpu_bf16.yaml}"
else
    ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-recipes/accelerate_configs/deepspeed_zero3_4gpu.yaml}"
fi
# Iteration-1 policy (Hub id). Llama example: POLICY_ITER1=meta-llama/Meta-Llama-3-8B-Instruct (accept license on HF).
POLICY_ITER1="${POLICY_ITER1:-mistralai/Mistral-7B-Instruct-v0.2}"
BASE_TOKENIZER="${BASE_TOKENIZER:-$POLICY_ITER1}"
# Rank 0 prints package versions + trl/transformers paths (train_nash_md --show_env). Clear PYTHONPATH if it
# includes .../RSPO/third_party/trl or you may load the wrong TRL tree.
# Default judge = local PairRM (no HTTPS to router.huggingface.co). Multi-GPU + PairRM often needs
# NUM_GPUS=1, or ALLOW_PAIRRM_MULTIGPU=1, or HF Inference: export HF_INFERENCE_JUDGE=1 and HF_TOKEN then JUDGE=pairrm.
# Use JUDGE=hf only if HTTPS/proxy to Hugging Face Inference works.
PAIRRM_DEVICE="${PAIRRM_DEVICE:-cpu}"
if [ "$NUM_GPUS" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
elif [ "$NUM_GPUS" -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
fi

iter_num=3
RUN_TAG="nashmd-promptABC"

for i in $(seq 1 $iter_num); do
    echo "Running Nash-MD Iter ${i}"
    if [ "$i" -eq 1 ]; then
        MODEL="$POLICY_ITER1"
        TOK_ARG=()
    else
        MODEL="checkpoints/${RUN_TAG}-Mistral-7B-Instruct-NashMD-Iter$((i-1))"
        TOK_ARG=(--tokenizer_name_or_path "$BASE_TOKENIZER")
    fi
    OUTPUT_DIR="checkpoints/${RUN_TAG}-Mistral-7B-Instruct-NashMD-Iter${i}"
    PROMPT_NUM=$(( (i - 1) % 3 + 1 ))
    PROMPT_DATASET="UCLA-AGI/data-mistral-7b-instruct-sppo-iter${PROMPT_NUM}"

    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        --num_processes "$NUM_GPUS" \
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
        --max_length 2048 \
        --judge_backend "$JUDGE" \
        --pairrm_device "$PAIRRM_DEVICE"
done
