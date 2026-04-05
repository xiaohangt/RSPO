#!/usr/bin/env python3
"""
Train with TRL NashMDTrainer (trl v1.0+ experimental) on SPPO-style scored data.

Expects the same `train.parquet` as SPPO (`chosen` / `rejected` as chat message lists).
Only the prompt (all turns before the last assistant message in `chosen`) is used;
Nash-MD generates completions online and scores with PairRMJudge.
"""
from __future__ import annotations

import argparse
import os

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl.experimental.judges import PairRMJudge
from trl.experimental.nash_md import NashMDConfig, NashMDTrainer


def load_sppo_style_dataset(dataset_arg: str, hub_org: str | None) -> Dataset:
    """Load train split from Hub (`org/name` or `name` with hub_org) or local `.../train.parquet`."""
    local_parquet = os.path.join(dataset_arg, "train.parquet")
    if os.path.isfile(local_parquet):
        ds = load_dataset("parquet", data_files=local_parquet, split="train")
    elif "/" in dataset_arg:
        ds = load_dataset(dataset_arg, split="train")
    elif hub_org:
        ds = load_dataset(f"{hub_org}/{dataset_arg}", split="train")
    else:
        ds = load_dataset(dataset_arg, split="train")
    return ds


def to_prompt_only(example: dict) -> dict:
    chosen = example["chosen"]
    return {"prompt": chosen[:-1]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Hub dataset repo id, or basename used with --hub_dataset_org, or local dir containing train.parquet",
    )
    p.add_argument(
        "--hub_dataset_org",
        type=str,
        default="UCLA-AGI",
        help="If --dataset has no '/', load_dataset(f'{org}/{dataset}', split='train')",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--learning_rate", type=float, default=5e-7)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--beta", type=float, default=0.001)
    p.add_argument("--mixture_coef", type=float, default=0.5)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_strategy", type=str, default="epoch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_sppo_style_dataset(args.dataset, args.hub_dataset_org if "/" not in args.dataset else None)
    cols = set(raw.column_names)
    if "chosen" not in cols:
        raise ValueError(f"Dataset must contain 'chosen' (got columns {raw.column_names})")
    drop = [c for c in raw.column_names if c != "chosen"]
    train_dataset = raw.map(to_prompt_only, remove_columns=drop)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    training_args = NashMDConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        beta=args.beta,
        mixture_coef=args.mixture_coef,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        seed=args.seed,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
    )

    judge = PairRMJudge()
    trainer = NashMDTrainer(
        model=model,
        ref_model=None,
        judge=judge,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
