#!/bin/bash
iter_num=3
# LOSS_TYPE=sppo_entropy
# LOSS_TYPE=sppo_reversekl
# LOSS_TYPE=sppo_forward
# LOSS_TYPE=sppo_chisq10
LOSS_TYPE=sppo_forwardimportance100
# LOSS_TYPE=sppo_forwardreverse
# LOSS_TYPE=sppo_forward1reverse50
# LOSS_TYPE=sppo
# REG_COEF=0.01
# REG_COEF=0
REG_COEF=0.1

for i in $(seq 1 $iter_num); do
    echo "Running Iter ${i}"
    if [ "$i" -eq 1 ]; then
        MODEL="mistralai/Mistral-7B-Instruct-v0.2"
    else
        MODEL="checkpoints/${LOSS_TYPE}-${REG_COEF}-PromptABC-Mistral-7B-Instruct-SPPO-Iter$((i-1))"
    fi
    OUTPUT_DIR="checkpoints/${LOSS_TYPE}-${REG_COEF}-PromptABC-Mistral-7B-Instruct-SPPO-Iter${i}"
    PROMPT_NUM=$(( (i - 1) % 3 + 1 ))
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter${PROMPT_NUM}"
    OUT="data-${LOSS_TYPE}-${REG_COEF}-promptABC-mistral-7b-instruct-sppo-iter${i}"
    DATASET_DIR="synthetic_data_${LOSS_TYPE}-${REG_COEF}-promptABC-mistral-7b-instruct-sppo-iter${i}_score"


    # if [ "$i" -ne 1 ]; then
    bash scripts/generate.sh --model $MODEL --prompt $PROMPT --out_path $OUT
    # fi
    bash scripts/pipeline_reg.sh --model $MODEL --iter $i \
    --dataset $DATASET_DIR \
    --output_dir $OUTPUT_DIR --num 1 --loss_type ${LOSS_TYPE} --reg_coef ${REG_COEF}
done
