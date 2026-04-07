#!/usr/bin/env python3
"""
Train with TRL NashMDTrainer (trl v1.0+ experimental).

Uses prompt-only data: Nash-MD samples completions online and scores with PairRMJudge.
Pass a Hub dataset with a string `prompt` column (e.g. UCLA-AGI/data-mistral-7b-instruct-sppo-iter{1,2,3}),
or SPPO-style `chosen`/`rejected` chat lists (only the prompt prefix from `chosen` is used).

Default policy: use ``models/Mistral-7B-Instruct-v0.2`` (env ``LOCAL_MISTRAL_7B``) when that directory exists;
otherwise ``mistralai/Mistral-7B-Instruct-v0.2`` from the Hub. Pass ``--model_name_or_path`` to override.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def mistral_7b_dir() -> str:
    """Default local Mistral snapshot under RSPO/models; override with env LOCAL_MISTRAL_7B."""
    v = os.environ.get("LOCAL_MISTRAL_7B")
    if v:
        return os.path.abspath(os.path.expanduser(v))
    return str(_REPO_ROOT / "models" / "Mistral-7B-Instruct-v0.2")


# When the default local snapshot is absent, load from Hub.
DEFAULT_MISTRAL_HUB = "mistralai/Mistral-7B-Instruct-v0.2"
from importlib.metadata import PackageNotFoundError, version as pkg_version

import logging

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

# Mistral (and others) trigger INFO "Could not locate ... custom_generate/generate.py" — optional Hub feature.
logging.getLogger("transformers.dynamic_module_utils").setLevel(logging.WARNING)

try:
    from trl.trainer.judges import HfPairwiseJudge, PairRMJudge
except ModuleNotFoundError:
    from trl.experimental.judges import HfPairwiseJudge, PairRMJudge


class HfPairwiseJudgeForNashMD(HfPairwiseJudge):
    """
    NashMDTrainer._compute_judge passes return_scores=True (soft probabilities for the first completion).
    TRL's HfPairwiseJudge.judge does not accept that argument; PairRMJudge does.
    """

    def judge(
        self,
        prompts: list[str],
        completions: list[list[str]],
        shuffle_order: bool = True,
        return_scores: bool = False,
        temperature: float = 1.0,
        **kwargs: object,
    ) -> list[int | float]:
        _ = temperature, kwargs  # API parity with PairRMJudge; HF API returns discrete 0/1
        ranks = super().judge(prompts, completions, shuffle_order=shuffle_order)
        if not return_scores:
            return ranks
        probs: list[float] = []
        for r in ranks:
            if r == 0:
                probs.append(1.0)
            elif r == 1:
                probs.append(0.0)
            else:
                probs.append(0.5)
        return probs


def _patch_trl_prepare_deepspeed_for_ref_model_batch() -> None:
    """
    OnlineDPOTrainer calls prepare_deepspeed() again for ref_model. The copied DeepSpeed config can keep a stale
    train_batch_size (e.g. from an accelerate yaml tuned for 4 GPUs) while the run uses --num_processes 2, which
    triggers: train_batch_size != micro_batch * grad_acc * world_size.
    Recompute train_batch_size from the plugin's micro batch + grad steps + current distributed world size.
    """
    import trl.models.utils as trl_model_utils

    def prepare_deepspeed(model, accelerator):
        import deepspeed  # lazy import, same as TRL

        deepspeed_plugin = accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
        stage = config_kwargs["zero_optimization"]["stage"]

        if model is not None:
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and stage == 3:
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )

        if dist.is_available() and dist.is_initialized():
            micro = config_kwargs.get("train_micro_batch_size_per_gpu")
            gas = config_kwargs.get("gradient_accumulation_steps")
            if isinstance(micro, int) and isinstance(gas, int):
                config_kwargs["train_batch_size"] = micro * gas * dist.get_world_size()

        if stage != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    trl_model_utils.prepare_deepspeed = prepare_deepspeed


_patch_trl_prepare_deepspeed_for_ref_model_batch()

from trl.experimental.nash_md import NashMDConfig, NashMDTrainer
from trl.import_utils import is_llm_blender_available


def _distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_log_process() -> bool:
    return _local_rank() == 0


def _suppress_verbose_library_logging_on_non_main() -> None:
    """Trainer logs 'Running training' on every rank; keep tqdm + rank-0 prints readable."""
    if _local_rank() == 0:
        return
    for name in (
        "transformers",
        "transformers.trainer",
        "accelerate",
        "datasets",
        "trl",
        "deepspeed",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def _warn_nash_md_if_no_cuda(args: argparse.Namespace) -> None:
    if torch.cuda.is_available() or not _is_log_process():
        return
    ws = _distributed_world_size()
    print(
        "\n"
        "train_nash_md: WARNING — PyTorch does not see a usable CUDA device (GPU usage will stay ~0%).\n"
        "  Nash-MD is online RL: each step runs generation (policy+ref mixture, up to 2× max_new_tokens per prompt)\n"
        "  plus the pairwise judge. On CPU with a 7B model this is extremely slow; the first step can take a long\n"
        "  time and progress may sit at 0% until generation and judging finish.\n"
        "  What to do:\n"
        "    • Fix the driver / CUDA stack so `torch.cuda.is_available()` is True, or install a PyTorch build\n"
        "      matching your driver (see https://pytorch.org/get-started/locally/).\n"
        "    • For a quick sanity check on CPU: NUM_GPUS=1, --per_device_train_batch_size 1, --max_new_tokens 32,\n"
        "      and consider DeepSpeed ZeRO-1/2 or no DeepSpeed instead of ZeRO-3.\n"
        f"  Current: judge_backend={args.judge_backend!r}, max_new_tokens={args.max_new_tokens}, "
        f"WORLD_SIZE={ws}.\n",
        flush=True,
    )


def _accelerate_mixed_precision() -> str:
    """Set by `accelerate launch` from the YAML `mixed_precision` field (e.g. bf16)."""
    return os.environ.get("ACCELERATE_MIXED_PRECISION", "").strip().lower()


def resolve_use_bf16(args: argparse.Namespace) -> bool:
    """
    Value passed into NashMDConfig(bf16=...). TrainingArguments.__post_init__ *rejects* bf16=True when
    transformers' is_torch_bf16_gpu_available() is False — even if DeepSpeed+Accelerate still use bf16 from the YAML.
    In that case we pass False here and fix up with sync_training_args_bf16_with_accelerate_deepspeed() after init.
    """
    accel_mp = _accelerate_mixed_precision()
    if accel_mp == "bf16":
        if not args.bf16:
            raise SystemExit(
                "train_nash_md: Accelerate mixed_precision is bf16 (see your accelerate yaml) but you passed --no-bf16. "
                "Either drop --no-bf16 or set mixed_precision to 'no' or 'fp16' in that yaml."
            )
        return bool(bf16_training_supported())
    return bool(args.bf16) and bf16_training_supported()


def _ds_config_bf16_enabled(cfg: dict) -> bool:
    block = cfg.get("bf16")
    if not isinstance(block, dict):
        return False
    v = block.get("enabled")
    if v is True:
        return True
    if isinstance(v, str) and v.lower() in ("true", "1", "yes", "on"):
        return True
    return False


def _ds_truthy(v: object) -> bool:
    if v is True:
        return True
    if isinstance(v, str) and v.strip().lower() in ("true", "1", "yes", "on"):
        return True
    return False


def deepspeed_plugin_wants_bf16(
    training_args: NashMDConfig,
    accelerator: object | None = None,
) -> bool:
    """Read Accelerate's DeepSpeed plugin (same source as transformers deepspeed_init / trainer_config_finalize)."""
    plugin = None
    if accelerator is not None:
        state = getattr(accelerator, "state", None)
        if state is not None:
            plugin = getattr(state, "deepspeed_plugin", None)
    if plugin is None:
        plugin = getattr(training_args, "deepspeed_plugin", None)
    if plugin is None:
        return False
    cfg = getattr(plugin, "deepspeed_config", None)
    if isinstance(cfg, dict) and _ds_config_bf16_enabled(cfg):
        return True
    hf_ds = getattr(plugin, "hf_ds_config", None)
    if hf_ds is None:
        return False
    try:
        v = hf_ds.get_value("bf16.enabled")
    except Exception:
        return False
    return _ds_truthy(v)


def sync_training_args_bf16_with_accelerate_deepspeed(
    training_args: NashMDConfig,
    cli_allows_bf16: bool,
    accelerate_mixed_precision_snapshot: str,
    accelerator: object | None = None,
) -> None:
    """
    HF `trainer_config_finalize` requires args.bf16 to match DeepSpeed's bf16.enabled.

    TrainingArguments may set bf16=False when is_torch_bf16_gpu_available() is False, then overwrite
    ACCELERATE_MIXED_PRECISION — so use a snapshot from the start of main *and* read the DeepSpeed plugin.
    """
    if not cli_allows_bf16:
        return
    if training_args.bf16 or getattr(training_args, "bf16_full_eval", False):
        return

    snap = accelerate_mixed_precision_snapshot.strip().lower()
    want_bf16 = snap in ("bf16", "fp8") or deepspeed_plugin_wants_bf16(training_args, accelerator)

    if not want_bf16:
        return

    if _is_log_process():
        print(
            "train_nash_md: DeepSpeed expects bf16 but TrainingArguments had bf16=False (HF GPU bf16 probe). "
            "Setting bf16=True so deepspeed_init can finalize.",
            flush=True,
        )
    training_args.bf16 = True


def _sync_deepspeed_train_batch_size_with_dist_world(trainer: object) -> None:
    """
    Trainer.propagate_args_to_deepspeed() may set train_batch_size with a stale world_size (e.g. 1), while
    torch.distributed already reflects the launched process count (e.g. 2). Accelerate's _prepare_deepspeed then
    computes the correct product but cannot overwrite a concrete train_batch_size (fill_match with must_match=False
    only fills 'auto'). DeepSpeed then asserts train_batch != micro * grad_acc * world_size.
    """
    if not getattr(trainer, "is_deepspeed_enabled", False):
        return
    accel = getattr(trainer, "accelerator", None)
    state = getattr(accel, "state", None) if accel is not None else None
    ds_plugin = getattr(state, "deepspeed_plugin", None) if state is not None else None
    if ds_plugin is None:
        return
    cfg = ds_plugin.deepspeed_config
    micro = cfg.get("train_micro_batch_size_per_gpu")
    gas = cfg.get("gradient_accumulation_steps")
    if not isinstance(micro, int) or not isinstance(gas, int):
        return

    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    want = micro * gas * world
    old = cfg.get("train_batch_size")
    if old == want:
        return
    cfg["train_batch_size"] = want
    if _is_log_process():
        print(
            f"train_nash_md: DeepSpeed train_batch_size {old!r} -> {want} "
            f"(train_micro_batch_size_per_gpu={micro}, gradient_accumulation_steps={gas}, world_size={world}).",
            flush=True,
        )


def need_explicit_ref_model_for_zero3() -> bool:
    """
    TRL builds ref_model via deepcopy when ref_model is None; that breaks DeepSpeed ZeRO-3.

    `is_deepspeed_zero3_enabled()` only becomes True after `TrainingArguments` constructs
    `HfTrainerDeepSpeedConfig` (e.g. `--deepspeed path.json`). With `accelerate launch` + a YAML
    config, DeepSpeed is injected through Accelerate and the HF weakref is never set, so we also
    read ACCELERATE_* env vars that launch sets before the script starts.
    """
    if is_deepspeed_zero3_enabled():
        return True
    if not _env_truthy("ACCELERATE_USE_DEEPSPEED"):
        return False
    raw = os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE", "").strip()
    if not raw:
        return False
    try:
        return int(raw) == 3
    except ValueError:
        return raw == "3"


def _pkg_ver(name: str) -> str:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return "not installed"


def _mod_file(name: str) -> str:
    try:
        return __import__(name).__file__ or "?"
    except Exception as e:
        return f"<import {name} failed: {e}>"


def _warn_vendor_trl_on_path() -> None:
    bad = [
        p
        for p in sys.path
        if os.path.normpath(p.rstrip("/")).endswith(os.path.join("third_party", "trl"))
    ]
    if bad:
        print(
            "train_nash_md: WARNING: sys.path contains RSPO/third_party/trl — that shadows pip/git TRL and causes "
            "subtle version bugs. Remove it from PYTHONPATH or delete that prefix entry.\n"
            f"  entries: {bad}",
            flush=True,
        )


def _path_for_display(p: str) -> str:
    """Show Hub ids as-is; only normalize real filesystem paths (avoids cwd/mistralai/... for org/name)."""
    exp = os.path.expanduser(p)
    if os.path.isabs(exp):
        return os.path.abspath(exp)
    candidate = os.path.abspath(exp)
    if os.path.exists(candidate):
        return candidate
    return p


def print_environment_diagnostic(args: argparse.Namespace) -> None:
    if not args.show_env or not _is_log_process():
        return
    raw_model = args.model_name_or_path
    raw_tok = args.tokenizer_name_or_path or args.model_name_or_path
    model_path = (
        os.path.abspath(os.path.expanduser(raw_model))
        if os.path.isabs(os.path.expanduser(raw_model)) or os.path.exists(os.path.abspath(os.path.expanduser(raw_model)))
        else None
    )
    tok_path = _path_for_display(raw_tok)
    print("=== train_nash_md environment (LOCAL_RANK=0) ===", flush=True)
    print(f"  executable: {sys.executable}", flush=True)
    if os.environ.get("PYTHONPATH"):
        print(f"  PYTHONPATH: {os.environ['PYTHONPATH']}", flush=True)
    _warn_vendor_trl_on_path()
    print(
        f"  torch {_pkg_ver('torch')} | transformers {_pkg_ver('transformers')} | trl {_pkg_ver('trl')} | "
        f"datasets {_pkg_ver('datasets')} | accelerate {_pkg_ver('accelerate')} | deepspeed {_pkg_ver('deepspeed')}",
        flush=True,
    )
    print(f"  trl import: {_mod_file('trl')}", flush=True)
    print(f"  transformers import: {_mod_file('transformers')}", flush=True)
    if is_llm_blender_available():
        print(f"  llm_blender {_pkg_ver('llm-blender')} | import: {_mod_file('llm_blender')}", flush=True)
    print(f"  policy model_name_or_path: {_path_for_display(raw_model)}", flush=True)
    print(f"  tokenizer path: {tok_path}", flush=True)
    if model_path is not None and os.path.isdir(model_path):
        cfg = os.path.join(model_path, "config.json")
        if os.path.isfile(cfg):
            with open(cfg, encoding="utf-8") as f:
                meta = json.load(f)
            print(
                f"  local config.json: model_type={meta.get('model_type')!r} "
                f"architectures={meta.get('architectures')!r}",
                flush=True,
            )
        else:
            print(f"  WARNING: local dir missing config.json: {cfg}", flush=True)
    print("=== end environment ===", flush=True)


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def resolve_judge_backend_for_launch(args: argparse.Namespace) -> None:
    """
    llm-blender PairRM builds a CrossCompareReranker whose weights load as size-0 tensors under
    `accelerate launch` / DeepSpeed multi-process (known interaction).

    Optional switch to HF Inference API only when explicitly requested: ``HF_INFERENCE_JUDGE=1`` plus a Hub
    token. That avoids forcing ``router.huggingface.co`` when proxies/SSL block outbound HTTPS.
    """
    if _distributed_world_size() <= 1:
        return
    if args.judge_backend != "pairrm":
        return
    if os.environ.get("ALLOW_PAIRRM_MULTIGPU") == "1" or args.allow_pairrm_multigpu:
        if _is_log_process():
            print(
                "train_nash_md: ALLOW_PAIRRM_MULTIGPU set — loading PairRM under multi-process (may crash).",
                flush=True,
            )
        return
    has_hf_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    if has_hf_token and _env_truthy("HF_INFERENCE_JUDGE"):
        if _is_log_process():
            print(
                "train_nash_md: multi-process + PairRM is unstable; HF_INFERENCE_JUDGE=1 + token → "
                "switching judge to hf (Inference API).",
                flush=True,
            )
        args.judge_backend = "hf"
        return
    raise SystemExit(
        "train_nash_md: With WORLD_SIZE>1, local PairRM (llm-blender) usually fails to load weights.\n"
        "  Pick one:\n"
        "  · NUM_GPUS=1 (single process; PairRM is reliable)\n"
        "  · ALLOW_PAIRRM_MULTIGPU=1 and --allow_pairrm_multigpu (may crash)\n"
        "  · HF_INFERENCE_JUDGE=1 plus HUGGINGFACE_HUB_TOKEN (or HF_TOKEN) — use HF Inference judge "
        "(needs HTTPS to router.huggingface.co; fix proxy or unset bad HTTP(S)_PROXY if calls fail)\n"
        "  · pass --judge_backend hf explicitly"
    )


class PairRMJudgeCPU(PairRMJudge):
    """Load PairRM on CPU (helps single-process runs; multi-process `accelerate launch` may still break)."""

    def __init__(self) -> None:
        if not is_llm_blender_available():
            raise ValueError("llm-blender is not installed. Install with: pip install llm-blender")
        import llm_blender

        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM", device=torch.device("cpu"))


def bf16_training_supported() -> bool:
    """TrainingArguments rejects bf16=True when there is no usable bf16 GPU."""
    if not torch.cuda.is_available():
        return False
    fn = getattr(torch.cuda, "is_bf16_supported", None)
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def _from_pretrained_local_kwargs(path: str) -> dict[str, bool]:
    """
    If `path` is an existing directory, force local load only.

    Otherwise transformers/huggingface_hub may treat absolute filesystem paths like repo ids and raise
    HFValidationError, or hit the network when offline / SSL fails.
    """
    exp = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(exp):
        return {"local_files_only": True}
    return {}


def _mistral_local_default_abspath() -> str:
    return os.path.abspath(os.path.expanduser(mistral_7b_dir()))


def _is_missing_mistral_local_default(path: str) -> bool:
    """True if `path` is the configured local Mistral root and that directory is not on disk."""
    exp = os.path.abspath(os.path.expanduser(path))
    return os.path.normpath(exp) == os.path.normpath(_mistral_local_default_abspath()) and not os.path.isdir(exp)


def resolve_model_and_tokenizer_paths(args: argparse.Namespace) -> None:
    """
    Default policy: use LOCAL_MISTRAL_7B / models/Mistral-7B-Instruct-v0.2 when present; otherwise Hub.

    If the user passes the same path explicitly (e.g. run_nash_md_mistral_ABC.sh) and the folder is still
    missing, fall back to Hub instead of exiting — matches \"install weights later\" workflows.
    """
    if args.model_name_or_path is None:
        if os.path.isdir(_mistral_local_default_abspath()):
            args.model_name_or_path = mistral_7b_dir()
        else:
            args.model_name_or_path = DEFAULT_MISTRAL_HUB
            if _is_log_process():
                print(
                    f"train_nash_md: no --model_name_or_path; local snapshot not found ({mistral_7b_dir()}); "
                    f"using Hub {DEFAULT_MISTRAL_HUB}.",
                    flush=True,
                )
    elif _is_missing_mistral_local_default(args.model_name_or_path):
        args.model_name_or_path = DEFAULT_MISTRAL_HUB
        if _is_log_process():
            print(
                f"train_nash_md: --model_name_or_path is the local Mistral default but directory is missing; "
                f"using Hub {DEFAULT_MISTRAL_HUB}.",
                flush=True,
            )

    if args.tokenizer_name_or_path and _is_missing_mistral_local_default(args.tokenizer_name_or_path):
        args.tokenizer_name_or_path = DEFAULT_MISTRAL_HUB
        if _is_log_process():
            print(
                f"train_nash_md: --tokenizer_name_or_path is the local Mistral default but directory is missing; "
                f"using Hub tokenizer {DEFAULT_MISTRAL_HUB}.",
                flush=True,
            )


def _require_existing_dir_if_absolute(path: str, label: str) -> None:
    """
    Require an existing directory only when ``path`` is a true filesystem path (absolute after ~ expansion).

    Hugging Face repo ids look like ``mistralai/Mistral-7B-Instruct-v0.2``; ``os.path.abspath`` would turn
    those into ``<cwd>/mistralai/...``, which must not be validated as a missing local directory.
    """
    exp = os.path.expanduser(path)
    if not os.path.isabs(exp):
        return
    exp = os.path.abspath(exp)
    if not os.path.isdir(exp):
        raise SystemExit(
            f"train_nash_md: {label} is not an existing directory:\n  {exp}\n"
            "  Download or symlink the model here, or pass a Hub id (e.g. mistralai/Mistral-7B-Instruct-v0.2)."
        )


def load_train_split(dataset_arg: str, hub_org: str | None) -> Dataset:
    """Load train split from Hub, local ``train.parquet``, or a directory saved with ``datasets.save_to_disk``."""
    expanded = os.path.abspath(os.path.expanduser(dataset_arg))
    if os.path.isdir(expanded):
        train_parquet = os.path.join(expanded, "train.parquet")
        if os.path.isfile(train_parquet):
            return load_dataset("parquet", data_files=train_parquet, split="train")
        try:
            from datasets import load_from_disk

            loaded = load_from_disk(expanded)
        except Exception as e:
            raise SystemExit(
                f"train_nash_md: could not load --dataset from directory:\n  {expanded}\n"
                "  Put train.parquet here, or save a dataset with datasets.Dataset.save_to_disk().\n"
                f"  Underlying error: {e}"
            ) from e
        if hasattr(loaded, "keys") and "train" in loaded:
            return loaded["train"]
        return loaded

    local_parquet = os.path.join(dataset_arg, "train.parquet")
    if os.path.isfile(local_parquet):
        return load_dataset("parquet", data_files=local_parquet, split="train")

    hub_id = dataset_arg if "/" in dataset_arg else (f"{hub_org}/{dataset_arg}" if hub_org else dataset_arg)

    try:
        return load_dataset(hub_id, split="train")
    except Exception as e:
        err = str(e).lower()
        if any(x in err for x in ("ssl", "connection", "max retries", "timed out", "unreachable")):
            raise SystemExit(
                f"train_nash_md: could not load dataset from the Hub ({hub_id}).\n"
                "  Fix network/SSL or use an offline copy: --dataset /abs/path/to/dir with train.parquet, "
                "or a directory from datasets.save_to_disk().\n"
                f"  Underlying error: {e}"
            ) from e
        raise


def to_prompt_from_chosen(example: dict) -> dict:
    chosen = example["chosen"]
    return {"prompt": chosen[:-1]}


def to_conversational_user_prompt(example: dict) -> dict:
    p = example["prompt"]
    if isinstance(p, str):
        conv = [{"role": "user", "content": p}]
    else:
        conv = p
    return {"prompt": conv}


def build_train_dataset(raw: Dataset) -> Dataset:
    cols = set(raw.column_names)
    if "prompt" in cols:
        return raw.map(
            to_conversational_user_prompt,
            remove_columns=[c for c in raw.column_names],
        )
    if "chosen" in cols:
        drop = [c for c in raw.column_names if c != "chosen"]
        return raw.map(to_prompt_from_chosen, remove_columns=drop)
    raise ValueError(
        "Dataset needs a `prompt` (string or chat messages) or `chosen` column; "
        f"got: {raw.column_names}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help=(
            "Policy checkpoint: local directory with config + weights, or a Hub model id. "
            "If omitted: use LOCAL_MISTRAL_7B / models/Mistral-7B-Instruct-v0.2 when that directory exists, "
            f"else {DEFAULT_MISTRAL_HUB} from the Hub."
        ),
    )
    p.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If set, load tokenizer from this local directory (else use --model_name_or_path)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Hub id, or local directory containing train.parquet, or a directory from datasets.save_to_disk() "
        "(no Hub access required).",
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
    p.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set for custom Hub models; off avoids custom_generate lookups on standard checkpoints (e.g. Mistral).",
    )
    p.add_argument(
        "--judge_backend",
        type=str,
        default="pairrm",
        choices=("pairrm", "hf"),
        help=(
            "pairrm: local llm-blender PairRM; hf: Hugging Face Inference API (HF_TOKEN; needs HTTPS to "
            "router.huggingface.co). Multi-GPU + pairrm: set HF_INFERENCE_JUDGE=1 to auto-switch to hf, "
            "or NUM_GPUS=1 / ALLOW_PAIRRM_MULTIGPU=1."
        ),
    )
    p.add_argument(
        "--pairrm_device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Where to load PairRM. Use cpu with DeepSpeed ZeRO-3 multi-GPU (default). cuda for single-GPU/no-DS.",
    )
    p.add_argument(
        "--show_env",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print versions, import paths, and local config.json summary (rank 0 only). Use --no-show_env to disable.",
    )
    p.add_argument(
        "--allow_pairrm_multigpu",
        action="store_true",
        default=False,
        help="Try PairRM under multi-GPU anyway (often crashes; use with ALLOW_PAIRRM_MULTIGPU=1).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    resolve_model_and_tokenizer_paths(args)
    _suppress_verbose_library_logging_on_non_main()
    # Capture before TrainingArguments / imports elsewhere can set ACCELERATE_MIXED_PRECISION to "no".
    accelerate_mp_snapshot = os.environ.get("ACCELERATE_MIXED_PRECISION", "").strip().lower()
    print_environment_diagnostic(args)
    resolve_judge_backend_for_launch(args)
    use_bf16 = resolve_use_bf16(args)
    if args.bf16 and not use_bf16 and _accelerate_mixed_precision() != "bf16":
        print(
            "train_nash_md: bf16 disabled (no CUDA or device lacks bf16); using fp32. "
            "Fix GPU/driver stack or pass --no-bf16 explicitly.",
            flush=True,
        )

    raw = load_train_split(args.dataset, args.hub_dataset_org if "/" not in args.dataset else None)
    train_dataset = build_train_dataset(raw)

    _require_existing_dir_if_absolute(args.model_name_or_path, "--model_name_or_path")
    if args.tokenizer_name_or_path:
        _require_existing_dir_if_absolute(args.tokenizer_name_or_path, "--tokenizer_name_or_path")

    tok_path = args.tokenizer_name_or_path or args.model_name_or_path
    _tok_kw = _from_pretrained_local_kwargs(tok_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path, trust_remote_code=args.trust_remote_code, **_tok_kw
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load PairRM *before* the policy when using DeepSpeed ZeRO-3 (`zero3_init_flag`): otherwise llm-blender's
    # CrossCompareReranker can be built under meta/empty init and `load_state_dict` fails (shape [0] vs checkpoint).
    judge: HfPairwiseJudgeForNashMD | PairRMJudgeCPU | PairRMJudge | None = None
    if args.judge_backend == "pairrm":
        if args.pairrm_device == "cpu":
            judge = PairRMJudgeCPU()
            if _is_log_process():
                print(
                    "train_nash_md: judge_backend=pairrm — local PairRM via llm-blender (loads llm-blender/PairRM; "
                    "no Inference API). Loaded before policy to avoid ZeRO-3 meta-init conflicts. --pairrm_device cpu.",
                    flush=True,
                )
        else:
            judge = PairRMJudge()
            if _is_log_process():
                print(
                    "train_nash_md: judge_backend=pairrm — local PairRM (loaded before policy). --pairrm_device cuda.",
                    flush=True,
                )

    # Build config before loading weights so `is_deepspeed_zero3_enabled()` can see `--deepspeed` if passed.
    cuda_ok = torch.cuda.is_available()
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
        bf16=use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        use_cpu=not cuda_ok,
        dataloader_pin_memory=cuda_ok,
    )
    _warn_nash_md_if_no_cuda(args)
    sync_training_args_bf16_with_accelerate_deepspeed(
        training_args,
        cli_allows_bf16=bool(args.bf16),
        accelerate_mixed_precision_snapshot=accelerate_mp_snapshot,
    )

    effective_bf16 = bool(training_args.bf16 or getattr(training_args, "bf16_full_eval", False))
    torch_dtype = torch.bfloat16 if effective_bf16 else torch.float32

    _model_kw = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": torch_dtype,
        **_from_pretrained_local_kwargs(args.model_name_or_path),
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **_model_kw)

    # ZeRO-3 shards parameters; deepcopy-based create_reference_model() is invalid. Load a second checkpoint copy.
    ref_model = None
    if need_explicit_ref_model_for_zero3():
        if _is_log_process():
            print(
                "train_nash_md: DeepSpeed ZeRO-3 — loading ref_model via from_pretrained "
                "(Trainer cannot deepcopy the policy).",
                flush=True,
            )
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **_model_kw)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    if args.judge_backend == "hf":
        _hf_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        judge = HfPairwiseJudgeForNashMD(token=_hf_tok)
        if _is_log_process():
            print(
                "train_nash_md: judge_backend=hf — Hugging Face Inference API (HTTPS to router.huggingface.co).",
                flush=True,
            )

    assert judge is not None

    trainer = NashMDTrainer(
        model=model,
        ref_model=ref_model,
        judge=judge,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
    )
    # TRL/Trainer init must not drop bf16=True; re-sync on the live args object right before deepspeed_init.
    sync_training_args_bf16_with_accelerate_deepspeed(
        trainer.args,
        cli_allows_bf16=bool(args.bf16),
        accelerate_mixed_precision_snapshot=accelerate_mp_snapshot,
        accelerator=getattr(trainer, "accelerator", None),
    )
    _sync_deepspeed_train_batch_size_with_dist_world(trainer)
    if _is_log_process():
        print(
            "train_nash_md: starting training (first step: generation + judge; can be slow on CPU or with large "
            "max_new_tokens).",
            flush=True,
        )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
