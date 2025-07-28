#!/bin/bash
# script for running the regularized SPPO with GEMMA-2-2B-IT

#### Best ReverseKL
LOSS_TYPE=sppo_reversekl
REG_COEF=0.5

#### Best ForwardKL
# LOSS_TYPE=sppo_forwardimportance7
# REG_COEF=0.1

#### Best ForwardKL + ReverseKL
# LOSS_TYPE=sppo_forward1reverse5
# REG_COEF=0.1
# LOSS_TYPE=sppo_forward1reverse50
# REG_COEF=0.01

iter_num=3
for i in $(seq 1 $iter_num); do
    echo "Running Iter ${i}"
    if [ "$i" -eq 1 ]; then
        MODEL="google/gemma-2-2b-it"
    else
        MODEL="checkpoints/${LOSS_TYPE}-${REG_COEF}-Gemma-2-2B-IT-RSPO-Iter$((i-1))"
    fi
    OUTPUT_DIR="checkpoints/${LOSS_TYPE}-${REG_COEF}-Gemma-2-2B-IT-RSPO-Iter${i}"
    PROMPT_NUM=$(( (i - 1) % 3 + 1 ))
    # PROMPT="UCLA-AGI/data-llama-3-8b-instruct-sppo-iter${PROMPT_NUM}"
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter${i}"
    OUT="data-${LOSS_TYPE}-${REG_COEF}-gemma-2-2b-it-rspo-iter${i}"
    DATASET_DIR="synthetic_data_${LOSS_TYPE}-${REG_COEF}-gemma-2-2b-it-rspo-iter${i}_score"
    
    # if [ "$i" -ne 1 ]; then
    bash scripts/generate.sh --model $MODEL --prompt $PROMPT --out_path $OUT
    # fi
    bash scripts/pipeline_reg.sh --model $MODEL --iter $i \
    --dataset $DATASET_DIR \
    --output_dir $OUTPUT_DIR --num 1 --loss_type ${LOSS_TYPE} --reg_coef ${REG_COEF}
done
