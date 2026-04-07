"""
Smoke test: Nash-MD + PairRM on a tiny slice of data (1 optimizer step).

Requires latest TRL from PyPI (`pip install -U trl[judges]` or `pip install -e .` with current setup.py).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from local_model_paths import qwen2_0_5b_instruct_dir

# Newer TRL: judges live under trl.trainer.judges; older / git layouts used trl.experimental.judges.
try:
    from trl.trainer.judges import PairRMJudge
except ModuleNotFoundError:
    from trl.experimental.judges import PairRMJudge

from trl.experimental.nash_md import NashMDConfig, NashMDTrainer


def main() -> None:
    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = qwen2_0_5b_instruct_dir()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    judge = PairRMJudge()

    raw = load_dataset("trl-lib/ultrafeedback-prompt", split="train")
    n = min(8, len(raw))
    raw = raw.select(range(n))
    # Match scripts/train_nash_md.py: Nash-MD expects conversational `prompt` messages.
    if "prompt" in raw.column_names:

        def to_conv(ex):
            p = ex["prompt"]
            conv = [{"role": "user", "content": p}] if isinstance(p, str) else p
            return {"prompt": conv}

        train_dataset = raw.map(to_conv, remove_columns=raw.column_names)
    else:
        train_dataset = raw

    training_args = NashMDConfig(
        output_dir="tmp_nashmd_smoke",
        max_steps=1,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
    )

    trainer = NashMDTrainer(
        model=model,
        ref_model=None,
        judge=judge,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
    )
    trainer.train()
    print("test.py: ok (1 training step)")


if __name__ == "__main__":
    main()
