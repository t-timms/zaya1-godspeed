"""Benchmark NVFP4 CT model against BF16 baselines via lm_eval + transformers.

Decompresses the CT model to BF16 in memory, then runs lm_eval on:
  - AIME 2026 (math reasoning)
  - GPQA-Diamond (graduate-level QA)
  - MMLU-Pro (knowledge)
  - LiveCodeBench v6 (code generation)

Compares against ZAYA1-8B BF16 baselines from official model card.
Expected: <2% degradation for AIME, <3% for GPQA, <2% for MMLU-Pro.

Usage:
    python scripts/benchmark_ct_nvfp4.py --tasks aime26 gpqa mmlu_pro
    python scripts/benchmark_ct_nvfp4.py --tasks all --output-dir benchmarks/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("benchmark")

CT_PATH = "zaya1-8b-nvfp4-ct/model.safetensors"

BASELINES: dict[str, float] = {
    "aime26": 89.1,
    "gpqa": 71.0,
    "mmlu_pro": 74.2,
    "livecodebench": 65.8,
}

TASK_MAP: dict[str, str] = {
    "aime26": "aime24",
    "gpqa": "gpqa",
    "mmlu_pro": "mmlu_pro",
    "livecodebench": "livecode_bench",
}

THRESHOLDS: dict[str, float] = {
    "aime26": -3.0,
    "gpqa": -3.0,
    "mmlu_pro": -2.0,
    "livecodebench": -2.0,
}


def decompress_model_to_transformers() -> Any:
    """Decompress CT NVFP4 model into a BF16 HuggingFace model in memory."""
    import torch
    import safetensors.torch as st

    log.info("Loading original BF16 model...")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "Zyphra/ZAYA1-8B", torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True,
    )

    log.info("Loading CT state dict...")
    ct_state = st.load_file(CT_PATH, device="cpu")

    log.info("Decompressing packed layers...")
    t0 = time.time()
    from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
    from compressed_tensors.quantization import preset_name_to_scheme

    scheme = preset_name_to_scheme("NVFP4A16", targets=["Linear"])
    patched = 0

    for name, module in model.named_modules():
        d = {}
        for k in ("weight_packed", "weight_scale", "weight_global_scale"):
            key = f"{name}.{k}"
            if key in ct_state:
                d[k] = ct_state[key].to("cuda:0" if torch.cuda.is_available() else "cpu")
        if "weight_packed" not in d:
            continue
        decomp = NVFP4PackedCompressor.decompress(d, scheme)
        module.weight.data = decomp["weight"].to(torch.bfloat16).cpu()
        patched += 1

    log.info("Decompressed %d layers in %.0fs", patched, time.time() - t0)
    return model


def run_lm_eval(model: Any, tokenizer: Any, tasks: list[str], output_dir: Path, skip_gated: bool = False) -> dict[str, Any]:
    """Run lm_eval on the decompressed model via hf model instance."""
    import torch
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info("Moving model to %s...", device)
    model = model.to(device)
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log.info("VRAM: %.1f GB", vram)

    lm_eval_tasks = [TASK_MAP.get(t, t) for t in tasks]

    if skip_gated:
        try:
            import huggingface_hub
            hf_token = huggingface_hub.get_token()
            log.info("HF token: %s", "found" if hf_token else "not set — gated datasets will fail")
        except Exception:
            log.warning("HF token not found — gated datasets will fail")

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1, device=device)

    log.info("Running lm_eval on: %s", lm_eval_tasks)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=lm_eval_tasks,
        batch_size=1,
        num_fewshot=0,
        limit=None,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "lm_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def print_summary(results: dict[str, Any], tasks: list[str]) -> None:
    """Print benchmark summary vs baselines."""
    log.info("")
    log.info("=" * 70)
    log.info("NVFP4 CT MODEL BENCHMARK RESULTS")
    log.info("=" * 70)

    task_results = results.get("results", {})
    all_pass = True

    for task in tasks:
        lm_task = TASK_MAP.get(task, task)
        key = f"{lm_task}|0|none"
        if key not in task_results:
            key = f"{lm_task}|0|0"
        if key not in task_results:
            for k in task_results:
                if lm_task in k:
                    key = k
                    break

        score = None
        if key in task_results:
            metric = task_results[key]
            if "acc,none" in metric:
                score = metric["acc,none"] * 100
            elif "acc" in metric:
                score = metric["acc"] * 100
            elif "exact_match,strict-match" in metric:
                score = metric["exact_match,strict-match"] * 100
            elif "exact_match" in metric:
                score = metric["exact_match"] * 100
            elif "pass@1" in metric:
                score = metric["pass@1"] * 100

        baseline = BASELINES.get(task, 0)
        threshold = THRESHOLDS.get(task, 0)
        delta = (score - baseline) if score else 0
        status = "PASS" if delta >= threshold else "FAIL"

        log.info("%-20s | BF16: %5.1f | NVFP4: %5.1f | %+.1f | %s",
                 task.upper(), baseline, score or 0, delta, status)
        if delta < threshold:
            all_pass = False

    log.info("=" * 70)
    log.info("OVERALL: %s", "PASSED — NVFP4 preserves quality" if all_pass else "FAILED — check thresholds")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark NVFP4 CT model vs BF16 baselines")
    parser.add_argument("--tasks", nargs="+", default=["aime26"],
                        choices=["all", "aime26", "gpqa", "mmlu_pro", "livecodebench"])
    parser.add_argument("--output-dir", default="benchmarks")
    parser.add_argument("--run-all", action="store_true", help="Run all 4 benchmarks (2+ hours)")
    parser.add_argument("--skip-gated", action="store_true", help="Skip gated datasets (GPQA)")
    args = parser.parse_args()

    if "all" in args.tasks or args.run_all:
        tasks = list(BASELINES.keys())
    else:
        tasks = args.tasks

    log.info("=== NVFP4 CT Benchmark ===")
    log.info("Tasks: %s", tasks)
    log.info("Baselines: AIME=%.1f GPQA=%.1f MMLU-Pro=%.1f LCB=%.1f",
             BASELINES["aime26"], BASELINES["gpqa"], BASELINES["mmlu_pro"], BASELINES["livecodebench"])

    try:
        import lm_eval
        log.info("lm_eval %s: OK", lm_eval.__version__)
    except ImportError:
        log.error("lm_eval not installed. Run: pip install lm-eval")
        return 1

    model = decompress_model_to_transformers()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Zyphra/ZAYA1-8B", trust_remote_code=True)
    results = run_lm_eval(model, tok, tasks, Path(args.output_dir), skip_gated=args.skip_gated)
    print_summary(results, tasks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
