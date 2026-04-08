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
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
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
def _generate_batch_transformers(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
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
    prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
    texts: list[str] = []
    for i, p_len in enumerate(prompt_lens):
        gen = out[i, int(p_len) :]
        texts.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return texts


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
        "--generation_batch_size",
        type=int,
        default=16,
        help="Batch size for generation requests. Increase if GPU memory allows.",
    )
    p.add_argument(
        "--backend",
        type=str,
        choices=("vllm", "transformers"),
        default="vllm",
        help="Generation backend. vllm is recommended for multi-GPU throughput.",
    )
    p.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
        help="vLLM tensor parallel size (set to number of GPUs, e.g. 8).",
    )
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.92,
        help="vLLM GPU memory utilization target.",
    )
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
    t0 = time.time()
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))

    tok_path = args.tokenizer_name_or_path or str(ckpt)
    generator_name = args.generator_name or ckpt.name

    out_json = Path(args.output_json) if args.output_json else ckpt / f"alpaca_eval2_outputs_{generator_name}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    print("=== AlpacaEval2: start ===", flush=True)
    print(f"checkpoint={ckpt}", flush=True)
    print(f"tokenizer={tok_path}", flush=True)
    print(f"generator_name={generator_name}", flush=True)
    print(f"device={args.device} dtype={args.dtype}", flush=True)
    print(
        f"gen_cfg: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, "
        f"top_p={args.top_p}, max_examples={args.max_examples}, "
        f"batch_size={args.generation_batch_size}",
        flush=True,
    )
    print(
        f"backend={args.backend}, tensor_parallel_size={args.tensor_parallel_size}, "
        f"gpu_memory_utilization={args.gpu_memory_utilization}",
        flush=True,
    )
    print(f"output_json={out_json}", flush=True)

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"resolved_device={device}", flush=True)

    print("[1/4] Loading tokenizer...", flush=True)
    t_tok = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tok_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[1/4] Tokenizer loaded in {time.time() - t_tok:.1f}s", flush=True)

    model = None
    llm = None
    sampling_params = None
    if args.backend == "transformers":
        print("[2/4] Loading model checkpoint with Transformers...", flush=True)
        t_model = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt),
            torch_dtype=torch_dtype if device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() and args.device.startswith("cuda") else None,
        )
        if not (torch.cuda.is_available() and args.device.startswith("cuda")):
            model.to(device)
        model.eval()
        print(f"[2/4] Model loaded in {time.time() - t_model:.1f}s", flush=True)
    else:
        print("[2/4] Loading model checkpoint with vLLM...", flush=True)
        t_model = time.time()
        from vllm import LLM, SamplingParams

        dtype_map = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
        llm = LLM(
            model=str(ckpt),
            tokenizer=tok_path,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=dtype_map[args.dtype],
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
        print(f"[2/4] vLLM engine ready in {time.time() - t_model:.1f}s", flush=True)

    print("[3/4] Loading AlpacaEval dataset...", flush=True)
    t_ds = time.time()
    eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    print(f"[3/4] Dataset loaded in {time.time() - t_ds:.1f}s (n={len(eval_set)})", flush=True)

    print("[4/4] Generating outputs...", flush=True)
    t_gen = time.time()
    outputs: list[dict[str, Any]] = []
    total = len(eval_set) if args.max_examples <= 0 else min(len(eval_set), args.max_examples)
    eval_list = [eval_set[i] for i in range(total)]
    pbar = tqdm(total=total, desc="Generating", unit="sample", dynamic_ncols=True)
    for start in range(0, total, args.generation_batch_size):
        end = min(start + args.generation_batch_size, total)
        batch = eval_list[start:end]
        prompts = [_build_prompt(tokenizer, ex["instruction"]) for ex in batch]

        if args.backend == "transformers":
            completions = _generate_batch_transformers(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        else:
            req_outputs = llm.generate(prompts, sampling_params)
            completions = [x.outputs[0].text.strip() for x in req_outputs]

        for ex, completion in zip(batch, completions):
            outputs.append(
                {
                    "instruction": ex["instruction"],
                    "output": completion,
                    "generator": generator_name,
                }
            )

        pbar.update(len(batch))
        pbar.set_postfix_str(f"last_out_chars={len(completions[-1]) if completions else 0}")

        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            cur = end
            if cur <= 3:
                print(
                    f"Sample {cur}: prompt_len={len(prompts[-1]) if prompts else 0} "
                    f"output_len={len(completions[-1]) if completions else 0} "
                    f"elapsed={time.time() - t_gen:.1f}s",
                    flush=True,
                )
            elif cur % max(10, args.generation_batch_size) == 0:
                per_item = (time.time() - t_gen) / cur
                eta = per_item * (total - cur)
                print(
                    f"Generated {cur}/{total} | {per_item:.2f}s/item | ETA {eta/60:.1f} min",
                    flush=True,
                )
    pbar.close()

    with out_json.open("w") as f:
        json.dump(outputs, f, ensure_ascii=False)

    print(
        f"[4/4] Generation complete in {time.time() - t_gen:.1f}s; "
        f"wrote {len(outputs)} outputs to {out_json}",
        flush=True,
    )

    if not args.run_alpaca_eval:
        print(f"=== Done in {time.time() - t0:.1f}s (generation only) ===", flush=True)
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
    t_eval = time.time()
    subprocess.run(cmd, check=True)
    print(f"AlpacaEval judging finished in {time.time() - t_eval:.1f}s", flush=True)
    print(f"=== Done in {time.time() - t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()

