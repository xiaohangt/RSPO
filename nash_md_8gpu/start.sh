export HF_TOKEN="${HF_TOKEN:-<set-your-hf-token>}"
export WANDB_API_KEY="${WANDB_API_KEY:-<set-your-wandb-api-key>}"


NUM_GPUS=8 bash run_nash_md_3iter_rspo.sh > log 2>&1
