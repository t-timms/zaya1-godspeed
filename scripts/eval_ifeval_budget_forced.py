"""Budget-forced IFEval for ZAYA1-8B W4A4 on 16 GB VRAM.

ZAYA1-8B is a long-reasoning model: the chat template opens the assistant turn with
`<think>` and the model rarely closes it within a feasible token budget. The stock
lm-eval `ifeval` task scores the FULL generation — including the un-terminated `<think>`
trace — against strict format rules, so a reasoning model scores far below its true
instruction-following ability (measured 32.5% vs Zyphra BF16 85.58%).

Fix, mirroring scripts/eval_gpqa_budget_forced.py:
  Stage 1  generate up to --think-budget reasoning tokens (stop early on </think>).
  Stage 2  inject "</think>\n\n" and generate the ACTUAL response.
  Score    run ONLY the Stage-2 response (the think trace stripped) through the official
           IFEval instruction checkers (vendored in lm_eval.tasks.ifeval).

Reports prompt-level and instruction-level, strict and loose accuracy — the same four
numbers as lm-eval/Zyphra. Zyphra BF16 reference: prompt_level_strict 85.58%.

Run (WSL, vllm-env active):
    python3 scripts/eval_ifeval_budget_forced.py --n 4          # smoke
    python3 scripts/eval_ifeval_budget_forced.py                # full 541
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")

MODEL = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-w4a4"
ZYPHRA_BF16_PROMPT_STRICT = 0.8558


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="limit to first N prompts (default: all 541)")
    ap.add_argument("--think-budget", type=int, default=2048, dest="think_budget")
    ap.add_argument("--response-max", type=int, default=2048, dest="response_max")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    ap.add_argument("--max-model-len", type=int, default=8192, dest="max_model_len")
    ap.add_argument("--output", default="results/ifeval_budget_forced.json")
    args = ap.parse_args()

    from datasets import load_dataset
    from lm_eval.tasks.ifeval.utils import InputExample, process_results
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("google/IFEval", split="train")
    if args.n is not None:
        ds = ds.select(range(min(args.n, len(ds))))
    docs = list(ds)

    base_prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": d["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for d in docs
    ]

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        moe_backend="cutlass",
        gpu_memory_utilization=0.92,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        enable_prefix_caching=False,
        kv_cache_dtype="fp8",
    )

    # Stage 1: bounded reasoning, stop early if the model closes </think> itself.
    sp_think = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.think_budget,
        stop=["</think>", "</s>"],
        seed=42,
    )
    think_outs = llm.generate(base_prompts, sp_think)
    closed = sum(o.outputs[0].stop_reason == "</think>" for o in think_outs)

    # Stage 2: force the end of reasoning, then generate the answer that gets scored.
    forced_prompts = [
        base + out.outputs[0].text + "\n</think>\n\n"
        for base, out in zip(base_prompts, think_outs)
    ]
    sp_ans = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.response_max,
        stop=["</s>", "<|im_end|>"],
        seed=42,
    )
    ans_outs = llm.generate(forced_prompts, sp_ans)

    # Score ONLY the post-think response with the official IFEval checkers.
    agg = {
        "prompt_level_strict_acc": [],
        "prompt_level_loose_acc": [],
        "inst_level_strict_acc": [],
        "inst_level_loose_acc": [],
    }
    per_prompt = []
    for d, a_out in zip(docs, ans_outs):
        response = a_out.outputs[0].text.strip()
        res = process_results(d, [response])
        agg["prompt_level_strict_acc"].append(bool(res["prompt_level_strict_acc"]))
        agg["prompt_level_loose_acc"].append(bool(res["prompt_level_loose_acc"]))
        agg["inst_level_strict_acc"].extend(res["inst_level_strict_acc"])
        agg["inst_level_loose_acc"].extend(res["inst_level_loose_acc"])
        per_prompt.append(
            {
                "key": d["key"],
                "instruction_id_list": d["instruction_id_list"],
                "prompt_strict": bool(res["prompt_level_strict_acc"]),
                "response_chars": len(response),
            }
        )

    def mean(xs: list) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    scores = {k: mean(v) for k, v in agg.items()}

    print("\n" + "=" * 72)
    print(f"BUDGET-FORCED IFEval  (n={len(docs)}, think_budget={args.think_budget})")
    print("=" * 72)
    print(f"{'metric':<26} {'ZAYA W4A4':>10} {'BF16 ref':>10}")
    print("-" * 72)
    print(f"{'prompt_level_strict_acc':<26} {scores['prompt_level_strict_acc'] * 100:>9.1f}% {ZYPHRA_BF16_PROMPT_STRICT * 100:>9.1f}%")
    print(f"{'prompt_level_loose_acc':<26} {scores['prompt_level_loose_acc'] * 100:>9.1f}% {'—':>10}")
    print(f"{'inst_level_strict_acc':<26} {scores['inst_level_strict_acc'] * 100:>9.1f}% {'—':>10}")
    print(f"{'inst_level_loose_acc':<26} {scores['inst_level_loose_acc'] * 100:>9.1f}% {'—':>10}")
    print("-" * 72)
    print(f"self-closed </think> within budget: {closed}/{len(docs)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "n": len(docs),
                "think_budget": args.think_budget,
                "scores": scores,
                "self_closed_think": closed,
                "bf16_prompt_strict_ref": ZYPHRA_BF16_PROMPT_STRICT,
                "per_prompt": per_prompt,
            },
            indent=2,
        )
    )
    print(f"Saved {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
