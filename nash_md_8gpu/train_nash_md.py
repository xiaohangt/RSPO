#!/usr/bin/env python3
"""
Nash-MD training (TRL experimental) — minimal script aligned with:
https://huggingface.co/docs/trl/nash_md_trainer

Multi-GPU PairRM: only **rank 0** loads PairRM (``PairRMJudgeRank0Only``); gather/scatter batches.
Use ``--pairrm_device cuda`` to put PairRM on the first GPU of rank 0 (``cuda:0``), or ``cpu`` to save VRAM.

Default backend is ``pairrm`` (set ``--preference_backend auto`` for reward on multi-GPU, or ``reward`` explicitly).

Examples:
  python train_nash_md.py
  accelerate launch ... train_nash_md.py --preference_backend pairrm --pairrm_device cuda
  python train_nash_md.py --preference_backend reward --reward_model trl-lib/Qwen2-0.5B-Reward
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from trl.experimental.judges import BasePairwiseJudge, PairRMJudge
from trl.experimental.nash_md import NashMDConfig, NashMDTrainer
from trl.trainer.utils import selective_log_softmax


def _patch_get_grad_fn_or_grad_acc_for_deepspeed_zero3() -> None:
    """
    PyTorch 2.6 + DeepSpeed ZeRO-3: ``count_used_parameters_in_backward`` calls
    ``torch.autograd.graph._get_grad_fn_or_grad_acc``. For some partitioned parameters,
    ``view_as(param).grad_fn`` can be None, which raises AttributeError inside PyTorch.
    DeepSpeed already skips ``None`` grad nodes; return None instead of crashing.
    """
    import torch.autograd.graph as ag

    if getattr(ag._get_grad_fn_or_grad_acc, "_nashmd_deepspeed_z3_patch", False):
        return

    _orig = ag._get_grad_fn_or_grad_acc

    def _safe_get_grad_fn_or_grad_acc(t):  # type: ignore[no-untyped-def]
        from torch.autograd.graph import GradientEdge

        if isinstance(t, GradientEdge):
            return t.node
        if t.requires_grad and t.grad_fn is None:
            with torch.enable_grad():
                v = t.view_as(t)
                if v.grad_fn is None:
                    return None
                node = v.grad_fn.next_functions[0][0]
        else:
            node = t.grad_fn
        return node

    _safe_get_grad_fn_or_grad_acc._nashmd_deepspeed_z3_patch = True  # type: ignore[attr-defined]
    ag._get_grad_fn_or_grad_acc = _safe_get_grad_fn_or_grad_acc


class NashMDTrainerZeRO3LogprobFix(NashMDTrainer):
    """
    ZeRO-3 + Nash-MD: logprob forward must use ``use_cache=False`` and leave inference mode so
    LlamaRMSNorm does not hit ``Inference tensors cannot be saved for backward``.
    """

    def _compute_logprobs(self, model, model_data, context_length):
        def compute_logprobs_for_data(m, data, need_grad: bool):
            with torch.inference_mode(False):
                if need_grad:
                    with torch.enable_grad():
                        output = m(
                            data["input_ids"],
                            attention_mask=data["attention_mask"],
                            use_cache=False,
                        )
                else:
                    with torch.no_grad():
                        output = m(
                            data["input_ids"],
                            attention_mask=data["attention_mask"],
                            use_cache=False,
                        )
            logits = output.logits[:, context_length - 1 : -1]
            return selective_log_softmax(logits, data["input_ids"][:, context_length:])

        model_logprobs_model_data = compute_logprobs_for_data(model, model_data, need_grad=True)

        with torch.no_grad():
            if self.ref_model is None:
                with model.disable_adapter():
                    ref_logprobs_model_data = compute_logprobs_for_data(model, model_data, need_grad=False)
            else:
                ref_logprobs_model_data = compute_logprobs_for_data(self.ref_model, model_data, need_grad=False)

        model_padding_mask = model_data["attention_mask"][:, context_length:] == 0
        model_logprobs_model_data = model_logprobs_model_data.masked_fill(model_padding_mask, 0.0)
        ref_logprobs_model_data = ref_logprobs_model_data.masked_fill(model_padding_mask, 0.0)

        return (model_logprobs_model_data, ref_logprobs_model_data)


def _tensor_is_inference(x: torch.Tensor) -> bool:
    fn = getattr(x, "is_inference", None)
    if not callable(fn):
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def _patch_llama_rmsnorm_inference_tensor_workaround() -> None:
    """
    ZeRO-3 + generate: activations can be inference tensors. Clone only when **grad is off** and the
    tensor is marked inference — same as stock ``LlamaRMSNorm`` when ``torch.is_grad_enabled()`` so
    DeepSpeed ZeRO-3 backward counting is not confused by extra ``clone()`` nodes during training.
    """
    try:
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
    except ImportError:
        return
    if getattr(LlamaRMSNorm.forward, "_nashmd_inference_tensor_patch", False):
        return

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled() and _tensor_is_inference(hidden_states):
            hidden_states = hidden_states.clone()
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        out = hidden_states.to(input_dtype)
        if not torch.is_grad_enabled() and _tensor_is_inference(out):
            out = out.clone()
        weight = self.weight
        if _tensor_is_inference(weight):
            weight = weight.clone()
        return weight * out

    forward._nashmd_inference_tensor_patch = True  # type: ignore[attr-defined]
    LlamaRMSNorm.forward = forward


try:
    from trl.import_utils import is_llm_blender_available
except ImportError:

    def is_llm_blender_available() -> bool:
        try:
            import llm_blender  # noqa: F401

            return True
        except ImportError:
            return False


def _ensure_llm_blender_importable() -> None:
    """Shim for transformers >= 5 (see TRL PairRMJudge)."""
    import transformers.utils.hub
    from packaging.version import Version

    if Version(transformers.__version__) >= Version("5.0.0"):
        transformers.utils.hub.TRANSFORMERS_CACHE = None


class PairRMJudgeCPU(BasePairwiseJudge):
    """PairRM on CPU (avoids extra VRAM on GPU 0; load before policy — see RSPO train_nash_md)."""

    def __init__(self) -> None:
        if not is_llm_blender_available():
            raise ValueError("llm-blender is not installed. Install with: pip install llm-blender")
        _ensure_llm_blender_importable()
        import llm_blender

        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM", device=torch.device("cpu"))

    def judge(
        self,
        prompts: list[str],
        completions: list[list[str]],
        shuffle_order: bool = True,
        return_scores: bool = False,
        temperature: float = 1.0,
    ) -> list[int | float]:
        return PairRMJudge.judge(self, prompts, completions, shuffle_order, return_scores, temperature)


class PairRMJudgeGPU(BasePairwiseJudge):
    """PairRM on the first CUDA device visible to this process (typically GPU 0 on rank 0)."""

    def __init__(self, device: torch.device | None = None) -> None:
        if not is_llm_blender_available():
            raise ValueError("llm-blender is not installed. Install with: pip install llm-blender")
        if not torch.cuda.is_available():
            raise ValueError("pairrm_device=cuda but CUDA is not available.")
        _ensure_llm_blender_importable()
        import llm_blender

        self.blender = llm_blender.Blender()
        dev = device or torch.device("cuda", torch.cuda.current_device())
        self.blender.loadranker("llm-blender/PairRM", device=dev)

    def judge(
        self,
        prompts: list[str],
        completions: list[list[str]],
        shuffle_order: bool = True,
        return_scores: bool = False,
        temperature: float = 1.0,
    ) -> list[int | float]:
        return PairRMJudge.judge(self, prompts, completions, shuffle_order, return_scores, temperature)


class PairRMJudgeRank0Only(BasePairwiseJudge):
    """
    Only rank 0 loads PairRM; other ranks gather prompts/completions to rank 0, rank 0 runs the judge,
    then results are scattered back. This avoids loading PairRM on every process and fixes local-checkpoint
    issues when only rank 0 has the Hub cache.
    """

    def __init__(self, pairrm_device: str) -> None:
        if not is_llm_blender_available():
            raise ValueError("llm-blender is not installed. Install with: pip install llm-blender")
        self._pairrm_device = pairrm_device
        self._rank = int(os.environ.get("RANK", "0"))
        self._world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self._inner: PairRMJudgeCPU | PairRMJudgeGPU | None = None
        if self._world_size == 1 or self._rank == 0:
            if pairrm_device == "cpu":
                self._inner = PairRMJudgeCPU()
            else:
                self._inner = PairRMJudgeGPU(torch.device("cuda", 0))

    def judge(
        self,
        prompts: list[str],
        completions: list[list[str]],
        shuffle_order: bool = True,
        return_scores: bool = False,
        temperature: float = 1.0,
    ) -> list[int | float]:
        if self._world_size == 1:
            assert self._inner is not None
            return self._inner.judge(prompts, completions, shuffle_order, return_scores, temperature)

        if not dist.is_initialized():
            raise RuntimeError(
                "PairRMJudgeRank0Only: torch.distributed is not initialized yet. "
                "The judge must run after the Trainer sets up the process group."
            )

        rank = dist.get_rank()
        ws = dist.get_world_size()
        obj = (prompts, completions, shuffle_order, return_scores, temperature)

        if rank == 0:
            gathered: list | None = [None] * ws
        else:
            gathered = None
        dist.gather_object(obj, object_gather_list=gathered, dst=0)

        scatter_list: list | None = None
        if rank == 0:
            assert gathered is not None and self._inner is not None
            all_prompts: list[str] = []
            all_completions: list[list[str]] = []
            for i in range(ws):
                p, c, _, _, _ = gathered[i]
                all_prompts.extend(p)
                all_completions.extend(c)
            sh0, rs0, t0 = gathered[0][2], gathered[0][3], gathered[0][4]
            out_full = self._inner.judge(
                all_prompts,
                all_completions,
                shuffle_order=sh0,
                return_scores=rs0,
                temperature=t0,
            )
            sizes = [len(gathered[i][0]) for i in range(ws)]
            chunks: list[list[int | float]] = []
            offset = 0
            for s in sizes:
                chunks.append(out_full[offset : offset + s])
                offset += s
            scatter_list = chunks

        recv_out: list[int | float | None] = [None]
        dist.scatter_object_list(recv_out, scatter_list, src=0)
        return recv_out[0]  # type: ignore[return-value]


def _distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _accelerate_deepspeed_zero3_env() -> bool:
    """True when `accelerate launch` + DeepSpeed ZeRO-3 (before Trainer may set HF deepspeed flags)."""
    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "").strip().lower() not in ("1", "true", "yes"):
        return False
    return os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE", "").strip() == "3"


def _need_explicit_ref_for_zero3() -> bool:
    """ZeRO-3 shards weights; Trainer must not build ref via deepcopy of the policy."""
    if is_deepspeed_zero3_enabled():
        return True
    return _accelerate_deepspeed_zero3_env()


def _sync_deepspeed_train_batch_size_with_dist_world(trainer: object) -> None:
    """
    Avoid DeepSpeed assert when train_batch_size in the plugin was filled with a stale world size.
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
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    if lr == 0:
        print(
            f"train_nash_md: DeepSpeed train_batch_size {old!r} -> {want} "
            f"(micro={micro}, grad_acc={gas}, world_size={world}).",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Nash-MD: PairRM (judge) or reward model — see --preference_backend."
    )
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base policy (Hub id or path). Default: Llama 3 8B Instruct (gated — accept license on HF).",
    )
    p.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="Tokenizer Hub id or path (defaults to --model_name_or_path). Use base model id when resuming from a saved checkpoint.",
    )
    p.add_argument(
        "--dataset",
        dest="dataset_name",
        type=str,
        default="trl-lib/ultrafeedback-prompt",
        help="Hub dataset id (prompt-only; e.g. trl-lib/ultrafeedback-prompt).",
    )
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--output_dir", type=str, default="Llama-3-8B-Instruct-NashMD")
    p.add_argument("--learning_rate", type=float, default=5e-7)
    p.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Micro-batch per GPU (DeepSpeed train_micro_batch_size_per_gpu). Use 1 if OOM in forward/backward or optimizer.step.",
    )
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Increase to match prior global batch when micro-batch is 1 (e.g. 1×8×8 GPUs ≈ 2×4×8).",
    )
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--beta", type=float, default=0.001)
    p.add_argument("--mixture_coef", type=float, default=0.5)
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Generation length per sample; lower if OOM.",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max sequence length for training batches (lower if OOM).",
    )
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Under DeepSpeed ZeRO-3, disabled by default (see NASHMD_KEEP_GRADIENT_CHECKPOINTING).",
    )
    p.add_argument(
        "--gradient_checkpointing_use_reentrant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass use_reentrant=True to gradient checkpointing. False matches TRL default but can break "
        "DeepSpeed ZeRO-3 + Mistral (Inference tensors cannot be saved for backward).",
    )
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--preference_backend",
        type=str,
        choices=("auto", "reward", "pairrm"),
        default="pairrm",
        help="pairrm (default): rank-0 PairRM. auto: reward if WORLD_SIZE>1 else pairrm. reward: sequence RM.",
    )
    p.add_argument(
        "--reward_model",
        type=str,
        default="trl-lib/Qwen2-0.5B-Reward",
        help="AutoModelForSequenceClassification, num_labels=1; must match policy tokenizer/chat template.",
    )
    p.add_argument(
        "--pairrm_device",
        type=str,
        choices=("cpu", "cuda"),
        default="cpu",
        help="PairRM device on rank 0: cpu (less VRAM) or cuda (first GPU, cuda:0). Multi-GPU: only rank 0 loads PairRM.",
    )
    return p.parse_args()


def _resolve_backend(args: argparse.Namespace) -> str:
    if args.preference_backend == "auto":
        return "reward" if _distributed_world_size() > 1 else "pairrm"
    return args.preference_backend


def main() -> None:
    args = parse_args()
    _patch_get_grad_fn_or_grad_acc_for_deepspeed_zero3()
    _patch_llama_rmsnorm_inference_tensor_workaround()
    backend = _resolve_backend(args)
    ws = _distributed_world_size()

    cuda_ok = torch.cuda.is_available()
    use_bf16 = bool(args.bf16) and cuda_ok

    train_dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    tok_path = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path,
        padding_side="left",
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

    judge = None
    reward_funcs = None

    # PairRM before policy weights: avoids llm-blender + DeepSpeed ZeRO-3 meta-init conflicts (RSPO order).
    # Multi-GPU: only rank 0 loads PairRM; gathers/scatters (PairRMJudgeRank0Only).
    if backend == "pairrm":
        if ws > 1:
            judge = PairRMJudgeRank0Only(args.pairrm_device)
        elif args.pairrm_device == "cpu":
            judge = PairRMJudgeCPU()
        else:
            judge = PairRMJudgeGPU()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    ref_model = None
    if _need_explicit_ref_for_zero3():
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    if backend == "reward":
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_model,
            num_labels=1,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )
        reward_model.eval()
        reward_funcs = [reward_model]

    # ZeRO-3 + gradient checkpointing + Mistral often raises:
    #   RuntimeError: Inference tensors cannot be saved for backward (RMSNorm)
    # even with use_reentrant=True. Default: turn off checkpointing under ZeRO-3 unless
    # NASHMD_KEEP_GRADIENT_CHECKPOINTING=1 (may still crash).
    effective_gradient_checkpointing = bool(args.gradient_checkpointing)
    if effective_gradient_checkpointing and _accelerate_deepspeed_zero3_env():
        if os.environ.get("NASHMD_KEEP_GRADIENT_CHECKPOINTING", "").strip().lower() not in (
            "1",
            "true",
            "yes",
        ):
            effective_gradient_checkpointing = False
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                print(
                    "train_nash_md: gradient checkpointing disabled under DeepSpeed ZeRO-3 "
                    "(Mistral + checkpoint + ZeRO-3 can hit 'Inference tensors cannot be saved for backward'). "
                    "Set NASHMD_KEEP_GRADIENT_CHECKPOINTING=1 to force it on. Uses more VRAM.",
                    flush=True,
                )

    gc_kwargs = None
    if effective_gradient_checkpointing and args.gradient_checkpointing_use_reentrant:
        gc_kwargs = {"use_reentrant": True}

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
        bf16=use_bf16,
        gradient_checkpointing=effective_gradient_checkpointing,
        gradient_checkpointing_kwargs=gc_kwargs,
        remove_unused_columns=False,
        use_cpu=not cuda_ok,
        dataloader_pin_memory=cuda_ok,
    )

    trainer = NashMDTrainerZeRO3LogprobFix(
        model=model,
        ref_model=ref_model,
        reward_funcs=reward_funcs,
        judge=judge,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
    )
    _sync_deepspeed_train_batch_size_with_dist_world(trainer)

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        extra = ""
        if backend == "pairrm":
            extra = f", pairrm_device={args.pairrm_device}"
        print(
            f"train_nash_md: preference_backend={backend} (WORLD_SIZE={ws}){extra}, "
            f"reward_model={args.reward_model if backend == 'reward' else 'n/a'}",
            flush=True,
        )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
