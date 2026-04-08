#!/usr/bin/env python3
"""
AlpacaEval 2.0 evaluation for a Hugging Face checkpoint.

This script has two phases:
1) Generate model outputs for the AlpacaEval eval set and write them to JSON.
2) (Optional) Run AlpacaEval judging via the `alpaca_eval` CLI.

Requirements:
  pip install alpaca_eval datasets transformers accelerate
  # plus a judge API key depending on annotators_config (e.g. export OPENAI_API_KEY=...)

Example:
  python eval_alpacaeval2.py \
    --checkpoint /path/to/output_dir/checkpoint-10 \
    --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --generator_name nash-md-iter1-ckpt10 \
    --max_new_tokens 512 \
    --run_alpaca_eval
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _build_prompt(tokenizer, instruction: str) -> str:
    # AlpacaEval instructions are single-turn. Prefer chat template if present.
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generate_prompt=True,
        )
    except Exception:
        # Fallback: plain instruction prompt.
        return instruction.strip() + "\n\n"


@torch.no_grad()
def _generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    do_sample = temperature > 0
    out = model.generate(
        **inputs,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return text.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate + judge AlpacaEval 2.0 for a checkpoint.")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="HF model checkpoint directory (e.g. .../output_dir/checkpoint-5 or final output_dir).",
    )
    p.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="Tokenizer to use (often the base model id). Defaults to --checkpoint.",
    )
    p.add_argument(
        "--generator_name",
        type=str,
        default=None,
        help="Name shown in AlpacaEval outputs (defaults to checkpoint folder name).",
    )
    p.add_argument("--device", type=str, default="cuda", help="cuda | cpu | cuda:0 etc.")
    p.add_argument("--dtype", type=str, default="bf16", choices=("fp16", "bf16", "fp32"))
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_examples", type=int, default=-1, help="Limit examples for a quick test.")
    p.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Where to write generated outputs JSON (defaults next to checkpoint).",
    )
    p.add_argument(
        "--run_alpaca_eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, run `alpaca_eval evaluate` after generation.",
    )
    p.add_argument(
        "--annotators_config",
        type=str,
        default="alpaca_eval_gpt4_turbo",
        help="AlpacaEval annotators config name (depends on your alpaca_eval install).",
    )
    p.add_argument(
        "--alpaca_eval_args",
        type=str,
        default="",
        help="Extra args passed to `alpaca_eval evaluate` (raw string).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))

    tok_path = args.tokenizer_name_or_path or str(ckpt)
    generator_name = args.generator_name or ckpt.name

    out_json = Path(args.output_json) if args.output_json else ckpt / f"alpaca_eval2_outputs_{generator_name}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(tok_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt),
        torch_dtype=torch_dtype if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

    outputs: list[dict[str, Any]] = []
    for i, ex in enumerate(eval_set):
        if args.max_examples > 0 and i >= args.max_examples:
            break
        instruction = ex["instruction"]
        prompt = _build_prompt(tokenizer, instruction)
        completion = _generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        # AlpacaEval expects at least these keys:
        # - instruction (or prompt), output, generator
        outputs.append(
            {
                "instruction": instruction,
                "output": completion,
                "generator": generator_name,
            }
        )

        if (i + 1) % 25 == 0 and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(f"Generated {i+1} / {len(eval_set)}", flush=True)

    with out_json.open("w") as f:
        json.dump(outputs, f, ensure_ascii=False)

    print(f"Wrote model outputs to {out_json}", flush=True)

    if not args.run_alpaca_eval:
        return

    # Run judging with the CLI to avoid coupling to alpaca_eval internal API variants.
    cmd = [
        "alpaca_eval",
        "evaluate",
        "--model_outputs",
        str(out_json),
        "--annotators_config",
        args.annotators_config,
    ]
    if args.alpaca_eval_args.strip():
        cmd.extend(args.alpaca_eval_args.strip().split())

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

