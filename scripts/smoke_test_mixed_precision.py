"""Quick smoke test for the mixed-precision W4A4 checkpoint.

Verifies:
  1. Checkpoint loads without errors
  2. Generates coherent text (not pad-token collapse)
  3. Reports which layers are BF16 vs W4A4 (from quantization manifest)

Run BEFORE the full benchmark suite to catch loading issues early.

Usage (WSL, vllm-env active):
    python3 scripts/smoke_test_mixed_precision.py [--model ./zaya1-8b-nvfp4-w4a4-soar]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-w4a4"


def print_manifest_summary(model_path: str) -> None:
    manifest_path = Path(model_path) / "quantization_manifest.json"
    if not manifest_path.exists():
        print("  [no manifest found]")
        return
    with open(manifest_path) as f:
        m = json.load(f)
    mp = m.get("mixed_precision", {})
    print(f"  Threshold:       {mp.get('threshold', 'N/A')}")
    print(f"  Outlier layers:  {mp.get('outlier_layers', [])}")
    print(f"  BF16 modules:    {mp.get('bf16_exempted_modules', 0)}")
    print(f"  W4A4 modules:    {mp.get('w4a4_compressed_modules', 0)}")
    print(f"  Description:     {mp.get('description', 'N/A')}")
    arc = m.get("arcquant", {})
    if arc.get("enabled"):
        print(f"  ARCQuant:        enabled, {len(arc.get('layers_corrected', []))} layers corrected")


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick smoke test for a W4A4 checkpoint")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to checkpoint directory (default: zaya1-8b-nvfp4-w4a4)",
    )
    args = parser.parse_args()
    model_path = args.model

    print("=" * 60)
    print("Mixed-Precision W4A4 Smoke Test")
    print("=" * 60)
    print(f"Model: {model_path}")

    print("\n[1] Manifest summary:")
    print_manifest_summary(model_path)

    print("\n[2] Loading model (enforce_eager for fast startup)...")
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vllm not importable — activate the env first:")
        print("  source /home/ttimm/vllm-env/bin/activate")
        return 1

    try:
        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            moe_backend="cutlass",
            enforce_eager=True,  # skip CUDA graph compile for quick smoke test
            gpu_memory_utilization=0.85,
            max_model_len=512,
            tensor_parallel_size=1,
            kv_cache_dtype="fp8",
        )
        print("  Model loaded OK")
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        return 1

    print("\n[3] Generation test (4 prompts):")
    prompts = [
        "The capital of France is",
        "In machine learning, a neural network is",
        "The tallest mountain in the world is",
        "Python is a programming language that",
    ]
    params = SamplingParams(temperature=0.0, max_tokens=24)
    try:
        outputs = llm.generate(prompts, params)
        all_ok = True
        for i, out in enumerate(outputs):
            text = out.outputs[0].text.strip()
            token_ids = out.outputs[0].token_ids
            is_collapse = len(set(token_ids)) <= 2  # pad-token collapse signal
            status = "COLLAPSE" if is_collapse else "OK"
            print(f"  [{status}] {prompts[i]!r:45} → {text!r}")
            if is_collapse:
                all_ok = False
        if not all_ok:
            print("\nWARNING: Token collapse detected — check global scale convention")
            return 2
    except Exception as e:
        print(f"  GENERATION FAILED: {e}")
        return 1

    print("\n[4] Speed test (128 tokens, no CUDA graphs):")
    speed_params = SamplingParams(temperature=0.0, max_tokens=128)
    t0 = time.time()
    out = llm.generate(["Explain the concept of quantization in deep learning:"], speed_params)
    elapsed = time.time() - t0
    n_tokens = len(out[0].outputs[0].token_ids)
    print(f"  {n_tokens} tokens in {elapsed:.1f}s = {n_tokens / elapsed:.1f} tok/s (enforce_eager, no graphs)")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED — proceed to: python3 scripts/run_full_benchmarks.py --model <path>")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
