#!/usr/bin/env bash
# Three Nash-MD iterations (RSPO-style chained checkpoints):
#   - Iter 1 trains from POLICY_ITER1; iter 2–3 continue from the previous checkpoint.
#   - One fixed dataset for all iterations (override DATASET / PROMPT_DATASET).
#
# Default base model: Meta Llama 3 8B Instruct (gated on HF — accept the license).
# RSPO reference: RSPO/run_nash_md_mistral_ABC.sh (same loop idea; defaults here use Llama).
#
# Requires: pip install "trl>=1.0" accelerate deepspeed llm-blender
#
# Default preference backend is pairrm (like RSPO JUDGE=pairrm). Multi-GPU: only rank 0 loads PairRM
# (gather/scatter); pairrm_device=cpu (default) or cuda (first GPU on rank 0). For a sequence RM: PREFERENCE_BACKEND=reward REWARD_MODEL=...
#
# See https://huggingface.co/docs/trl/nash_md_trainer

export OPENAI_API_KEY=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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_nash_md.py"

NUM_GPUS="${NUM_GPUS:-8}"
if [ "${NUM_GPUS}" -eq 1 ]; then
  ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${SCRIPT_DIR}/accelerate_configs/single_gpu_bf16.yaml}"
else
  ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${SCRIPT_DIR}/accelerate_configs/deepspeed_zero3_8gpu.yaml}"
fi

POLICY_ITER1="${POLICY_ITER1:-meta-llama/Meta-Llama-3-8B-Instruct}"
BASE_TOKENIZER="${BASE_TOKENIZER:-$POLICY_ITER1}"
RUN_TAG="${RUN_TAG:-nashmd-rspo-3iter-llama}"
PREFERENCE_BACKEND="${PREFERENCE_BACKEND:-pairrm}"
PAIRRM_DEVICE="${PAIRRM_DEVICE:-cpu}"

# Prompt-only data (override for UCLA / other Hub ids).
DATASET="${DATASET:-${PROMPT_DATASET:-trl-lib/ultrafeedback-prompt}}"

CKPT_ROOT="${CKPT_ROOT:-${SCRIPT_DIR}/checkpoints}"

if [ "${NUM_GPUS}" -eq 1 ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
elif [ "${NUM_GPUS}" -eq 2 ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
fi

# With PREFERENCE_BACKEND=reward and NUM_GPUS>1, REWARD_MODEL must be set.
_need_reward_model=0
if [ "${NUM_GPUS}" -gt 1 ]; then
  case "${PREFERENCE_BACKEND}" in
    reward) _need_reward_model=1 ;;
    auto) _need_reward_model=1 ;;
    *) _need_reward_model=0 ;;
  esac
fi
if [ "${_need_reward_model}" -eq 1 ] && [ -z "${REWARD_MODEL:-}" ]; then
  echo "run_nash_md_3iter_rspo: PREFERENCE_BACKEND=${PREFERENCE_BACKEND} with NUM_GPUS=${NUM_GPUS} requires REWARD_MODEL." >&2
  echo "  Or use PairRM: PREFERENCE_BACKEND=pairrm (default) and pip install llm-blender." >&2
  exit 1
fi

iter_num=3

for i in $(seq 1 "${iter_num}"); do
  echo "=== Nash-MD iteration ${i} / ${iter_num} (dataset=${DATASET}) ==="
  if [ "${i}" -eq 1 ]; then
    MODEL="${POLICY_ITER1}"
    TOK_ARG=()
  else
    MODEL="${CKPT_ROOT}/${RUN_TAG}-NashMD-Iter$((i - 1))"
    TOK_ARG=(--tokenizer_name_or_path "${BASE_TOKENIZER}")
  fi
  OUTPUT_DIR="${CKPT_ROOT}/${RUN_TAG}-NashMD-Iter${i}"

  EXTRA_ARGS=(--preference_backend "${PREFERENCE_BACKEND}")
  if [ -n "${REWARD_MODEL:-}" ]; then
    EXTRA_ARGS+=(--reward_model "${REWARD_MODEL}")
  fi
  if [ "${PREFERENCE_BACKEND}" = "pairrm" ]; then
    EXTRA_ARGS+=(--pairrm_device "${PAIRRM_DEVICE}")
  fi

  ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-info}" accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    --main_process_port "$((${MAIN_PROCESS_PORT_BASE:-29331} + i))" \
    "${TRAIN_SCRIPT}" \
    --model_name_or_path "${MODEL}" \
    "${TOK_ARG[@]}" \
    --dataset "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
    --learning_rate "${LEARNING_RATE:-5e-7}" \
    --beta "${BETA:-0.001}" \
    --mixture_coef "${MIXTURE_COEF:-0.5}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
    --max_length "${MAX_LENGTH:-1024}" \
    "$@"
done

echo "Done. Final checkpoint: ${CKPT_ROOT}/${RUN_TAG}-NashMD-Iter${iter_num}"
