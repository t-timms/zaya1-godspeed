#!/usr/bin/env python3
"""Gate a CUDA-graph mode on OUTPUT CORRECTNESS before anyone measures its speed.

This exists because of RESEARCH.md 5.14: a 102.6 / 407.4 tok/s figure was
published from a configuration that generated numerically wrong output. The
speed number was real; the tokens were garbage. Any throughput claim for this
model must therefore be gated on a coherence check run in the SAME engine
configuration, not a separate "looks fine" spot check.

Protocol
--------
Greedy (temperature=0) generation on a fixed prompt set, compared against a
reference captured under `--mode NONE` (enforce_eager), which is the only
configuration confirmed to generate correctly on SM120.

Different execution modes dispatch different kernels, so bit-exact agreement is
not expected even at temperature 0. The gate is therefore similarity-based, with
absolute garbage checks that fail regardless of similarity:

    PASS          mean similarity >= 0.90 and no garbage check tripped
    INCONCLUSIVE  0.60 <= mean similarity < 0.90  (needs a human read)
    FAIL          mean similarity < 0.60, or any garbage check tripped

Usage
-----
    # once, to capture the reference
    python scripts/check_coherence.py --model M --mode NONE --write-reference

    # then per candidate mode
    python scripts/check_coherence.py --model M --mode PIECEWISE

Exit code is 0 only on PASS. sweep_cudagraph_modes.sh relies on that.
"""
from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import re
import sys

MODES = ["NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"]

# Fixed, deterministic, and deliberately varied: a factual continuation, a
# code completion, an arithmetic chain, and a long-context-ish recall task.
# Kept short so the gate costs seconds, not minutes.
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "Compute step by step: 17 * 24 = ",
    "Q: Name three primary colors.\nA:",
    "The following is a list of the planets in our solar system, in order from the sun:",
]

PASS_THRESHOLD = 0.90
INCONCLUSIVE_THRESHOLD = 0.60


def garbage_checks(text: str) -> list[str]:
    """Absolute failure conditions, independent of the reference."""
    tripped = []
    if not text.strip():
        # Deliberately vague on cause: empty text can mean an immediate stop OR a
        # full budget of pad tokens. The per-prompt [gate] line printed by
        # generate() carries finish_reason and token-id counts, which is what
        # actually distinguishes them. Do not guess here.
        tripped.append("empty output (check the [gate] line for finish_reason)")
        return tripped

    printable = sum(1 for c in text if c.isprintable() or c in "\n\t")
    if printable / len(text) < 0.95:
        tripped.append(f"non-printable ratio {1 - printable / len(text):.2%}")

    # A single token repeated many times in a row is the classic corrupted-kernel
    # signature (pad-token collapse, repeated punctuation).
    if re.search(r"(.{1,12}?)\1{15,}", text):
        tripped.append("degenerate repetition (>=16x repeated span)")

    words = text.split()
    if len(words) >= 20 and len(set(words)) / len(words) < 0.15:
        tripped.append(f"vocabulary collapse (unique/total {len(set(words)) / len(words):.2%})")

    return tripped


def generate(model: str, mode: str, gpu_mem: float, max_tokens: int) -> list[str]:
    from vllm import LLM, SamplingParams

    kwargs: dict[str, object] = dict(
        model=model,
        dtype="bfloat16",
        kv_cache_dtype="fp8",
        gpu_memory_utilization=gpu_mem,
        max_model_len=4096,
        # REQUIRED: ZAYA's CCA state is not cacheable and vLLM defaults this on.
        enable_prefix_caching=False,
        trust_remote_code=True,
    )
    if mode == "NONE":
        kwargs["enforce_eager"] = True
    else:
        kwargs["compilation_config"] = {"cudagraph_mode": mode}

    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0)
    outs = llm.generate(PROMPTS, params)

    # Report the DISCRIMINATING detail, not just the decoded text. Empty text is
    # ambiguous: finish_reason="stop" with no text means the model chose to stop,
    # while finish_reason="length" with no text means it emitted a full budget of
    # tokens that decode to nothing - i.e. pad-token collapse, the signature of a
    # wrong NVFP4 global-scale convention (RESEARCH.md 5.9). Those have opposite
    # causes, and a gate that prints only the text invites the wrong story.
    for o in outs:
        gen = o.outputs[0]
        ids = list(gen.token_ids)
        uniq = set(ids)
        collapsed = len(uniq) == 1 and len(ids) > 4
        print(
            f"  [gate] n_tok={len(ids):>4} finish={gen.finish_reason!r:>10} "
            f"distinct_ids={len(uniq):>4} text_len={len(gen.text):>5}"
            + (f"  <-- SINGLE-TOKEN COLLAPSE (id={ids[0]})" if collapsed else "")
        )

    by_prompt = {o.prompt: o.outputs[0].text for o in outs}
    return [by_prompt[p] for p in PROMPTS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--reference", default="results/cudagraph_sweep/reference.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-reference", action="store_true")
    args = ap.parse_args()

    ref_path = pathlib.Path(args.reference)
    out_path = pathlib.Path(args.out or f"results/cudagraph_sweep/coherence_{args.mode}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate the reference BEFORE loading a model. Discovering a missing
    # reference after a full engine init wastes a load and the VRAM with it.
    if not args.write_reference and not ref_path.exists():
        print(f"no reference at {ref_path}; run once with --mode NONE --write-reference",
              file=sys.stderr)
        return 2

    texts = generate(args.model, args.mode, args.gpu_mem, args.max_tokens)

    if args.write_reference:
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(
            json.dumps({"model": args.model, "mode": args.mode, "outputs": texts}, indent=2),
            encoding="utf-8",
        )
        print(f"reference written -> {ref_path} (mode={args.mode})")
        # Still garbage-check the reference itself; a corrupt reference would
        # silently bless every later mode.
        tripped = [t for txt in texts for t in garbage_checks(txt)]
        if tripped:
            print("REFERENCE IS NOT CLEAN: " + "; ".join(tripped), file=sys.stderr)
            return 1
        print("verdict: PASS (reference, garbage checks clean)")
        return 0

    ref = json.loads(ref_path.read_text(encoding="utf-8"))["outputs"]

    sims, all_tripped, per_prompt = [], [], []
    for prompt, got, want in zip(PROMPTS, texts, ref):
        sim = difflib.SequenceMatcher(None, want, got).ratio()
        tripped = garbage_checks(got)
        sims.append(sim)
        all_tripped.extend(tripped)
        per_prompt.append(
            {"prompt": prompt, "similarity": round(sim, 4), "garbage": tripped,
             "output": got[:400]}
        )

    mean_sim = sum(sims) / len(sims)
    if all_tripped:
        verdict = "FAIL"
    elif mean_sim >= PASS_THRESHOLD:
        verdict = "PASS"
    elif mean_sim >= INCONCLUSIVE_THRESHOLD:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"

    result = {
        "mode": args.mode,
        "model": args.model,
        "verdict": verdict,
        "mean_similarity": round(mean_sim, 4),
        "min_similarity": round(min(sims), 4),
        "garbage_checks_tripped": all_tripped,
        "per_prompt": per_prompt,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"mode={args.mode} verdict={verdict} mean_sim={mean_sim:.4f} min_sim={min(sims):.4f}")
    if all_tripped:
        print("  garbage: " + "; ".join(sorted(set(all_tripped))))
    print(f"  -> {out_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
