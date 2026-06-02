#!bin/bash

# base model
#python3 eval_selfbleu.py mistralai/Mistral-7B-Instruct-v0.2 2024
# python3 eval_selfbleu.py mistralai/Mistral-7B-Instruct-v0.2 2025
# python3 eval_selfbleu.py mistralai/Mistral-7B-Instruct-v0.2 2026
# python3 eval_selfbleu.py mistralai/Mistral-7B-Instruct-v0.2 2027

# Vanilla SPPO
# python3 eval_selfbleu.py UCLA-AGI/Mistral7B-PairRM-SPPO-Iter1 2024
# python3 eval_selfbleu.py UCLA-AGI/Mistral7B-PairRM-SPPO-Iter2 2024
# python3 eval_selfbleu.py UCLA-AGI/Mistral7B-PairRM-SPPO-Iter3 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo-0-PromptABC-Mistral-7B-Instruct-SPPO-Iter1 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo-0-PromptABC-Mistral-7B-Instruct-SPPO-Iter2 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo-0-PromptABC-Mistral-7B-Instruct-SPPO-Iter3 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo-0-PromptABC-Mistral-7B-Instruct-SPPO-Iter4 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo-0-PromptABC-Mistral-7B-Instruct-SPPO-Iter5 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo-0-PromptABC-Mistral-7B-Instruct-SPPO-Iter6 2024

# # reverseKL
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_reversekl-0.5-PromptABC-Mistral-7B-Instruct-SPPO-Iter1 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_reversekl-0.5-PromptABC-Mistral-7B-Instruct-SPPO-Iter2 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_reversekl-0.5-PromptABC-Mistral-7B-Instruct-SPPO-Iter3 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_reversekl-0.5-PromptABC-Mistral-7B-Instruct-SPPO-Iter4 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_reversekl-0.5-PromptABC-Mistral-7B-Instruct-SPPO-Iter5 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_reversekl-0.5-PromptABC-Mistral-7B-Instruct-SPPO-Iter6 2024

# forwardKL + forwardreverse
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_forward1reverse5-0.1-PromptABC-Mistral-7B-Instruct-SPPO-Iter1 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_forward1reverse5-0.1-PromptABC-Mistral-7B-Instruct-SPPO-Iter2 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_forward1reverse5-0.1-PromptABC-Mistral-7B-Instruct-SPPO-Iter3 2024

# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_forwardimportance10-0.1-PromptABC-Mistral-7B-Instruct-SPPO-Iter1 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_forwardimportance10-0.1-PromptABC-Mistral-7B-Instruct-SPPO-Iter2 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_forwardimportance10-0.1-PromptABC-Mistral-7B-Instruct-SPPO-Iter3 2024

# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_forwardimportance10-0.5-PromptABC-Mistral-7B-Instruct-SPPO-Iter1 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_forwardimportance10-0.5-PromptABC-Mistral-7B-Instruct-SPPO-Iter2 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_forwardimportance10-0.5-PromptABC-Mistral-7B-Instruct-SPPO-Iter3 2024


# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_forward1reverse5-0.1-PromptABC-Mistral-7B-Instruct-SPPO-Iter1 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_forward1reverse5-0.1-PromptABC-Mistral-7B-Instruct-SPPO-Iter2 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_forward1reverse5-0.1-PromptABC-Mistral-7B-Instruct-SPPO-Iter3 2024

# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_chisq10-0.1-PromptABC-Mistral-7B-Instruct-SPPO-Iter1 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_chisq10-0.1-PromptABC-Mistral-7B-Instruct-SPPO-Iter2 2024
# python3 eval_selfbleu.py RegularizedSelfPlay/sppo_chisq10-0.1-PromptABC-Mistral-7B-Instruct-SPPO-Iter3 2024

python3 eval_selfbleu.py RegularizedSelfPlay/sppo_reversekl-0.05-PromptABC-LLAMA-3-8B-Instruct-SPPO-Iter1 2024
python3 eval_selfbleu.py RegularizedSelfPlay/sppo_reversekl-0.05-PromptABC-LLAMA-3-8B-Instruct-SPPO-Iter2 2024
python3 eval_selfbleu.py RegularizedSelfPlay/sppo_reversekl-0.05-PromptABC-LLAMA-3-8B-Instruct-SPPO-Iter3 2024





