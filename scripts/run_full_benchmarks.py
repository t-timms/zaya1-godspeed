"""Benchmark suite aligned to Zyphra's published ZAYA1-8B evaluation table.

Tasks run via lm-eval + vLLM (lm-eval 0.4.12):
  gpqa_diamond  → leaderboard_gpqa_diamond   0-shot   448 q    BF16: 71.0%
  mmlu_pro      → leaderboard_mmlu_pro       5-shot   12k q    BF16: 74.2%
  ifeval        → leaderboard_ifeval         0-shot   541 q    BF16: 85.58%

Requires separate evaluation harnesses (not included here):
  LiveCodeBench-v6  BF16: 65.8%   — https://livecodebench.github.io/
  BFCL-v4          BF16: 39.22%  — https://gorilla.cs.berkeley.edu/leaderboard.html
  τ²               BF16: 43.12%  — https://github.com/xlang-ai/tau-bench

Estimated runtime: GPQA ~10 min + MMLU-Pro ~50 min + IFEval ~20 min ≈ 80 min.
For a quick 30-min spot check, use:  --tasks gpqa_diamond ifeval

Hardware: RTX 5070 Ti SM120 — CUTLASS NVFP4 + CUDA graphs.

Usage (from WSL, vllm-env active):
    python3 scripts/run_full_benchmarks.py --model ./zaya1-8b-nvfp4-w4a4-mrgptq-v2

    # Quick spot check (skip the slow MMLU-Pro):
    python3 ... --tasks gpqa_diamond ifeval

    # Limit examples per task for fast iteration:
    python3 ... --limit 100
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_MODEL = (
    "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/"
    "zaya1-8b-nvfp4-w4a4"
)

# Zyphra's published BF16 numbers — the ceiling we're compressing toward.
# Source: Zyphra/ZAYA1-8B model card.
BF16_REFERENCE: dict[str, float] = {
    "gpqa_diamond": 0.710,   # GPQA-Diamond
    "mmlu_pro":     0.742,   # MMLU-Pro
    "ifeval":       0.8558,  # IFEval prompt-level-strict
    # Separate harnesses — not run here:
    # "livecodebench_v6": 0.658,
    # "bfcl_v4":          0.3922,
    # "tau2":             0.4312,
}

# Short CLI name → (lm_eval_task_name, num_fewshot_override, primary_metric)
# num_fewshot=None → use the task's built-in default (leaderboard tasks already encode this).
TASKS: dict[str, tuple[str, int | None, str]] = {
    "gpqa_diamond": ("leaderboard_gpqa_diamond", None, "acc_norm"),
    "mmlu_pro":     ("leaderboard_mmlu_pro",     None, "acc"),
    "ifeval":       ("leaderboard_ifeval",        None, "prompt_level_strict_acc"),
}


def build_model_args(
    model_path: str,
    gpu_mem: float = 0.99,
    enforce_eager: bool = False,
) -> str:
    args = (
        f"pretrained={model_path},"
        "dtype=bfloat16,"
        "moe_backend=cutlass,"
        f"gpu_memory_utilization={gpu_mem},"
        "max_model_len=4096,"
        "tensor_parallel_size=1,"
        "kv_cache_dtype=fp8"
    )
    if enforce_eager:
        args += ",enforce_eager=True"
    return args


def _get_metric(task_res: dict, metric_prefix: str) -> float | None:
    """Extract metric value, trying 'metric,none' then 'metric' key forms."""
    for key in (f"{metric_prefix},none", metric_prefix):
        val = task_res.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    # IFEval also uses acc_norm as a fallback key in some lm-eval versions
    if metric_prefix == "prompt_level_strict_acc":
        for alt in ("acc_norm,none", "acc_norm", "acc,none", "acc"):
            val = task_res.get(alt)
            if isinstance(val, (int, float)):
                return float(val)
    # GPQA acc_norm fallback to acc
    if metric_prefix == "acc_norm":
        for alt in ("acc,none", "acc"):
            val = task_res.get(alt)
            if isinstance(val, (int, float)):
                return float(val)
    return None


def print_results_table(
    task_results: dict[str, dict],
    elapsed: float,
    model_path: str,
) -> None:
    print()
    print("=" * 88)
    print("ZAYA1-8B  NVFP4 W4A4  vs  BF16 Reference (Zyphra published)")
    print("=" * 88)
    header = (
        f"{'Task':<16} {'lm-eval name':<30} {'W4A4':>8}  "
        f"{'BF16':>8}  {'Gap':>7}  {'Retained':>9}"
    )
    print(header)
    print("-" * 88)

    for short_name, (lm_task, _, metric_prefix) in TASKS.items():
        task_res = task_results.get(short_name, {})
        val = _get_metric(task_res, metric_prefix)
        bf16 = BF16_REFERENCE.get(short_name)

        val_str  = f"{val  * 100:.1f}%" if val  is not None else "—"
        bf16_str = f"{bf16 * 100:.1f}%" if bf16 is not None else "N/A"

        if val is not None and bf16 is not None:
            gap_str = f"{(val - bf16) * 100:+.1f}%"
            ret_str = f"{val / bf16 * 100:.1f}%"
        else:
            gap_str = "—"
            ret_str = "—"

        print(
            f"{short_name:<16} {lm_task:<30} {val_str:>8}  "
            f"{bf16_str:>8}  {gap_str:>7}  {ret_str:>9}"
        )

    print("=" * 88)
    print()
    print("Not evaluated here (require separate harnesses):")
    print("  LiveCodeBench-v6  BF16 65.8%   livecodebench package")
    print("  BFCL-v4           BF16 39.22%  gorilla BFCL framework")
    print("  τ²                BF16 43.12%  xlang-ai/tau-bench")
    print()
    print(f"Model:   {model_path}")
    print(f"Runtime: {elapsed / 60:.1f} min")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark ZAYA1-8B W4A4 against Zyphra's published BF16 numbers"
    )
    parser.add_argument(
        "--model", "--model-path",
        default=DEFAULT_MODEL,
        dest="model",
        help=f"Path to checkpoint (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        default="results/lmeval_w4a4_zyphra.json",
        help="Output JSON path (relative to project dir or absolute)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit examples per task for quick iteration",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.99,
        dest="gpu_mem",
        help="vLLM gpu_memory_utilization (default: 0.99)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        dest="enforce_eager",
        help="Disable CUDA graphs (slower but lower peak memory)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS.keys()),
        choices=list(TASKS.keys()),
        help=(
            "Tasks to run (default: all). "
            "Quick 30-min run: gpqa_diamond ifeval"
        ),
    )
    args = parser.parse_args()

    try:
        from lm_eval import simple_evaluate
    except ImportError:
        print("ERROR: lm_eval not installed. Run: pip install lm-eval")
        return 1

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login

            login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass

    model_args = build_model_args(args.model, args.gpu_mem, args.enforce_eager)
    all_results: dict[str, dict] = {}
    t_start = time.time()

    for short_name in args.tasks:
        lm_task, n_shot_override, metric_prefix = TASKS[short_name]
        shot_str = f"{n_shot_override}-shot" if n_shot_override is not None else "task-default"
        print(f"\n--- Running {short_name} ({lm_task}, {shot_str}) ---")
        t0 = time.time()
        try:
            kwargs: dict = dict(
                model="vllm",
                model_args=model_args,
                tasks=[lm_task],
                batch_size="auto",
                device="cuda",
                random_seed=42,
                numpy_random_seed=42,
            )
            if n_shot_override is not None:
                kwargs["num_fewshot"] = n_shot_override
            if args.limit is not None:
                kwargs["limit"] = args.limit

            result = simple_evaluate(**kwargs)
            task_res = result.get("results", {}).get(lm_task, {})
            all_results[short_name] = task_res
            elapsed_task = time.time() - t0
            val = _get_metric(task_res, metric_prefix)
            val_str = f"{val * 100:.1f}%" if val is not None else "?"
            print(f"  {short_name}: {metric_prefix}={val_str}  ({elapsed_task / 60:.1f} min)")
        except Exception as e:
            print(f"  ERROR running {short_name}: {e}")
            all_results[short_name] = {}
        finally:
            # Release the vLLM engine before starting the next task — without
            # this the CUDA graph pool stays resident and the next init OOMs.
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    elapsed_total = time.time() - t_start
    print_results_table(all_results, elapsed_total, args.model)

    combined = {
        "model": args.model,
        "results": all_results,
        "bf16_reference": BF16_REFERENCE,
        "elapsed_seconds": elapsed_total,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=2))
    print(f"Full results saved to {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
