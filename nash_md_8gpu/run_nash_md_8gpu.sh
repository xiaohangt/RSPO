#!/usr/bin/env bash
# Nash-MD on 8 GPUs: Accelerate + DeepSpeed ZeRO-3 by default (shards weights — avoids ZeRO-2 OOM on 8B).
#
# Default: PairRM (see train_nash_md.py). Use PREFERENCE_BACKEND=reward + REWARD_MODEL for a sequence RM.
#
# Requires: pip install "trl>=1.0" accelerate deepspeed
# Optional: llm-blender (only if you use PairRM)
#
# https://huggingface.co/docs/trl/nash_md_trainer

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_nash_md.py"

NUM_GPUS="${NUM_GPUS:-8}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${SCRIPT_DIR}/accelerate_configs/deepspeed_zero3_8gpu.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# Reduces allocator fragmentation (optional; override by exporting PYTORCH_CUDA_ALLOC_CONF before running).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
DATASET="${DATASET:-trl-lib/ultrafeedback-prompt}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/checkpoints/nash-md-8gpu-run}"
# Checkpointing (Hugging Face Trainer): steps | epoch | no
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-5}"
PREFERENCE_BACKEND="${PREFERENCE_BACKEND:-pairrm}"

EXTRA_LAUNCH=(--preference_backend "${PREFERENCE_BACKEND}" --pairrm_device "${PAIRRM_DEVICE:-cpu}")
if [ "${PREFERENCE_BACKEND}" = "reward" ]; then
  REWARD_MODEL="${REWARD_MODEL:?Set REWARD_MODEL when PREFERENCE_BACKEND=reward}"
  EXTRA_LAUNCH+=(--reward_model "${REWARD_MODEL}")
fi

# ZeRO-3 default YAML uses CPU optimizer offload (slower steps, avoids Adam-step OOM on ~80GB).
# GPU optimizer (faster): ACCELERATE_CONFIG="${SCRIPT_DIR}/accelerate_configs/deepspeed_zero3_8gpu_no_offload.yaml"
# ZeRO-2 (replicates full weights/GPU, often OOM on 8B): .../deepspeed_zero2_8gpu.yaml
# ZeRO-2 + CPU optimizer offload: .../deepspeed_zero2_8gpu_optimizer_cpu.yaml
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "NUM_GPUS=${NUM_GPUS}  ACCELERATE_CONFIG=${ACCELERATE_CONFIG}"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}  PREFERENCE_BACKEND=${PREFERENCE_BACKEND}"
echo "DATASET=${DATASET}  OUTPUT_DIR=${OUTPUT_DIR}"
echo "SAVE_STRATEGY=${SAVE_STRATEGY}  SAVE_STEPS=${SAVE_STEPS}"

ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-info}" accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29331}" \
  "${TRAIN_SCRIPT}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset "${DATASET}" \
  --output_dir "${OUTPUT_DIR}" \
  "${EXTRA_LAUNCH[@]}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning_rate "${LEARNING_RATE:-5e-7}" \
  --beta "${BETA:-0.001}" \
  --mixture_coef "${MIXTURE_COEF:-0.5}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
  --max_length "${MAX_LENGTH:-1024}" \
  --save_strategy "${SAVE_STRATEGY}" \
  --save_steps "${SAVE_STEPS}" \
  "$@"
