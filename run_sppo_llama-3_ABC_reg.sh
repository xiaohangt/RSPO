#!/bin/bash
# script for running the regularized SPPO with LLAMA-3-8B-Instruct

#### Best ReverseKL
LOSS_TYPE=sppo_reversekl
REG_COEF=0.5

#### Best ForwardKL
# LOSS_TYPE=sppo_forwardimportance10
# REG_COEF=0.1

#### Best ForwardKL + ReverseKL
# LOSS_TYPE=sppo_forward1reverse5
# REG_COEF=0.1

iter_num=3
for i in $(seq 1 $iter_num); do
    echo "Running Iter ${i}"
    if [ "$i" -eq 1 ]; then
        MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
    else
        MODEL="checkpoints/${LOSS_TYPE}-${REG_COEF}-Llama-3-8B-Instruct-RSPO-Iter$((i-1))"
    fi
    OUTPUT_DIR="checkpoints/${LOSS_TYPE}-${REG_COEF}-Llama-3-8B-Instruct-RSPO-Iter${i}"
    PROMPT_NUM=$(( (i - 1) % 3 + 1 ))
    # PROMPT="UCLA-AGI/data-llama-3-8b-instruct-sppo-iter${PROMPT_NUM}"
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter${i}"
    OUT="data-${LOSS_TYPE}-${REG_COEF}-llama-3-8b-instruct-rspo-iter${i}"
    DATASET_DIR="synthetic_data_${LOSS_TYPE}-${REG_COEF}-llama-3-8b-instruct-rspo-iter${i}_score"

    bash scripts/generate.sh --model $MODEL --prompt $PROMPT --out_path $OUT
    bash scripts/pipeline_reg.sh --model $MODEL --iter $i \
    --dataset $DATASET_DIR \
    --output_dir $OUTPUT_DIR --num 1 --loss_type ${LOSS_TYPE} --reg_coef ${REG_COEF}
done
