"""Phase A: paired loglikelihood suite for the 9.46 GB vs 6.02 GB W4A4 checkpoints.

ARC-Easy alone was weak evidence for "no measurable cost" (n=2376, +1.81 pp at
p=0.18, with acc_norm moving the other way). These four tasks are all pure
loglikelihood - no generation - so they are cheap per sample, give far more
statistical power for the same wall clock, and are structurally immune to the
`<think>`-never-terminates artifact that invalidated the earlier GPQA/IFEval
numbers.

WHY PER-DOC RECORDS ARE MANDATORY HERE
--------------------------------------
Comparing two runs' aggregate accuracies is an UNPAIRED two-proportion test -
the same weak test that produced ARC-Easy's p=0.18. The pairing that gives this
phase its power only exists if per-item outcomes are retained and the discordant
pairs are tested (McNemar). Two quantizations of one base model agree on the
overwhelming majority of items; that correlation is the entire reason to run
this. So `log_samples=True` is not optional, and every run writes a compact
per-doc JSONL alongside the aggregate JSON. Without it the night produces
numbers that cannot answer the question, and cannot be salvaged without re-running.

Deliberately NO chat template, unlike scripts/run_full_benchmarks.py. That flag
exists because IFEval and GPQA are generative tasks on an instruct model. These
four are ranked-continuation loglikelihood tasks, where the continuation must
follow the context directly - wrapping it in a chat turn changes what is scored
and breaks comparability with published numbers. What matters for Phase A is
that BOTH checkpoints see identical settings.

One task per process, on purpose: a single OOM or CUDA fault then costs one task
rather than the whole overnight run, and VRAM is fully reclaimed between tasks.

Usage (from WSL, vllm-env active):
    python3 scripts/run_phase_a.py --model ./zaya1-8b-nvfp4-w4a4-arcbase \
                                   --label 6.02GB --task hellaswag
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Must be set before vLLM is imported: the CUDA-graph estimator reserves 3.5+ GiB
# on a 16 GB card and pushes the KV cache into OOM.
os.environ.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PHASE_A_TASKS = ("hellaswag", "arc_challenge", "winogrande", "piqa")


def build_model_args(model_path: str, gpu_mem: float) -> str:
    # enable_prefix_caching=False is not optional: vLLM 0.20.2 defaults it True,
    # which trips zaya.py's assertion because CCA state is not cacheable.
    return (
        f"pretrained={model_path},"
        "dtype=bfloat16,"
        "moe_backend=cutlass,"
        f"gpu_memory_utilization={gpu_mem},"
        "max_model_len=4096,"
        "tensor_parallel_size=1,"
        "kv_cache_dtype=fp8,"
        "enforce_eager=True,"
        "enable_prefix_caching=False"
    )


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compact_sample(rec: dict) -> dict:
    """Reduce one lm-eval sample to what a paired test needs.

    Keeps the per-choice loglikelihoods rather than only the graded outcome, so
    `acc` can be re-derived independently (argmax over lls) as a check on the
    harness. Note `acc_norm` canNOT be recomputed from these fields - it divides
    by each continuation's byte length, which is not stored - so lm-eval's
    per-doc `acc_norm` grade is kept alongside; that grade is what the paired
    test consumes. Drops `doc` and `arguments`, which dominate the size.
    """
    lls: list[float] = []
    is_greedy: list[bool] = []
    for resp in rec.get("filtered_resps") or rec.get("resps") or []:
        item = resp[0] if isinstance(resp, (list, tuple)) and resp else resp
        if isinstance(item, (list, tuple)):
            ll = _as_float(item[0])
            if ll is not None:
                lls.append(ll)
            if len(item) > 1:
                is_greedy.append(bool(item[1]))
        else:
            ll = _as_float(item)
            if ll is not None:
                lls.append(ll)

    out: dict[str, Any] = {
        "doc_id": rec.get("doc_id"),
        "target": rec.get("target"),
        "lls": lls,
    }
    if is_greedy:
        out["is_greedy"] = is_greedy
    for metric in ("acc", "acc_norm"):
        if metric in rec:
            out[metric] = _as_float(rec[metric])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="checkpoint path")
    ap.add_argument("--label", required=True, help="short id used in output filenames")
    ap.add_argument("--task", required=True, choices=PHASE_A_TASKS)
    ap.add_argument("--outdir", default="results/phase_a")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test only")
    args = ap.parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        print(f"FATAL: checkpoint not found: {model_path}", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{args.label}__{args.task}.json"
    samples_out = outdir / f"{args.label}__{args.task}.samples.jsonl"

    # Resume only if BOTH artifacts exist - an aggregate without its per-doc
    # records is useless for the paired test and must be redone.
    if out.exists() and samples_out.exists():
        print(f"SKIP (already done): {out}")
        return 0

    from lm_eval import simple_evaluate  # imported late so env vars above apply

    print(f"[{time.strftime('%H:%M:%S')}] START {args.label} / {args.task}", flush=True)
    t0 = time.time()

    results = simple_evaluate(
        model="vllm",
        model_args=build_model_args(str(model_path), args.gpu_mem),
        tasks=[args.task],
        num_fewshot=0,
        batch_size="auto",
        device="cuda",
        limit=args.limit,
        random_seed=42,
        numpy_random_seed=42,
        fewshot_random_seed=42,
        log_samples=True,
    )

    elapsed = time.time() - t0
    res = (results or {}).get("results", {}).get(args.task, {})
    samples = (results or {}).get("samples", {}).get(args.task, [])

    if not samples:
        print(
            "FATAL: lm-eval returned no per-doc samples - the paired test cannot "
            "be run and this result would be misleading. Refusing to write.",
            file=sys.stderr,
        )
        return 3

    with samples_out.open("w") as fh:
        for rec in samples:
            fh.write(json.dumps(compact_sample(rec)) + "\n")

    payload = {
        "label": args.label,
        "task": args.task,
        "model": str(model_path),
        "elapsed_sec": round(elapsed, 1),
        "n_samples": len(samples),
        "settings": {
            "num_fewshot": 0,
            "apply_chat_template": False,
            "log_samples": True,
            "seeds": 42,
            "limit": args.limit,
            "model_args": build_model_args(str(model_path), args.gpu_mem),
        },
        "metrics": res,
        "samples_file": samples_out.name,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))

    print(
        f"[{time.strftime('%H:%M:%S')}] DONE  {args.label} / {args.task}  "
        f"acc={res.get('acc,none')}  acc_norm={res.get('acc_norm,none')}  "
        f"n={len(samples)}  ({elapsed / 60:.1f} min)",
        flush=True,
    )
    print(f"  -> {out}", flush=True)
    print(f"  -> {samples_out} ({samples_out.stat().st_size / 1048576:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
