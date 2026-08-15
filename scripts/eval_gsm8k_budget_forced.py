"""Budget-forced GSM8K for ZAYA1-8B W4A4 on 16 GB VRAM.

Same problem as GPQA/IFEval: ZAYA1-8B's <think> trace does not reliably close
within a feasible token budget, so a stock zero-shot-CoT harness either times
out mid-reasoning or scores the unterminated trace. This mirrors
scripts/eval_gpqa_budget_forced.py:
  Stage 1  generate up to --think-budget reasoning tokens (stop early on </think>).
  Stage 2  inject "</think>\n\nThe answer is " and decode a short numeric answer.
  Score    reuse lm-eval's own gsm8k-cot-zeroshot extraction regexes so the
           number-parsing logic matches the standard harness exactly.

Run (WSL, vllm-env active):
    python3 scripts/eval_gsm8k_budget_forced.py --n 20         # smoke
    python3 scripts/eval_gsm8k_budget_forced.py                # full 1319
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")

DEFAULT_MODEL = "./zaya1-8b-nvfp4-w4a4-arcbase"

# Same regex lm-eval's gsm8k-cot-zeroshot flexible-extract filter uses, applied
# to our own forced short answer instead of a free-running generation.
_FLEXIBLE_EXTRACT = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def extract_number(text: str) -> str | None:
    matches = list(_FLEXIBLE_EXTRACT.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    raw = m.group(0)
    return raw.replace(",", "").replace("$", "").rstrip(".")


def gold_answer(answer_field: str) -> str:
    # GSM8K gold answers are "<reasoning> #### <number>"; lm-eval strips
    # everything up to and including "#### " and any trailing period/commas.
    tail = answer_field.split("#### ")[-1].strip()
    return tail.replace(",", "").rstrip(".")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint path")
    ap.add_argument("--n", type=int, default=None, help="limit to first N problems (default: all 1319)")
    ap.add_argument("--think-budget", type=int, default=2048, dest="think_budget")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    ap.add_argument("--max-model-len", type=int, default=8192, dest="max_model_len")
    ap.add_argument("--output", default="results/gsm8k_budget_forced.json")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.n is not None:
        ds = ds.select(range(min(args.n, len(ds))))
    docs = list(ds)

    base_prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": f"{d['question']}\nLet's think step by step."}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for d in docs
    ]

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        moe_backend="cutlass",
        gpu_memory_utilization=0.92,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        enable_prefix_caching=False,
        kv_cache_dtype="fp8",
    )

    sp_think = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.think_budget,
        stop=["</think>", "</s>"],
        seed=42,
    )
    think_outs = llm.generate(base_prompts, sp_think)
    closed = sum(o.outputs[0].stop_reason == "</think>" for o in think_outs)

    forced_prompts = [
        base + out.outputs[0].text + "\n</think>\n\nThe answer is " for base, out in zip(base_prompts, think_outs)
    ]
    sp_ans = SamplingParams(temperature=0.0, max_tokens=32, stop=["\n", "</s>"], seed=42)
    ans_outs = llm.generate(forced_prompts, sp_ans)

    correct = 0
    rows = []
    for d, t_out, a_out in zip(docs, think_outs, ans_outs):
        gold = gold_answer(d["answer"])
        pred = extract_number(a_out.outputs[0].text)
        ok = pred is not None and pred == gold
        correct += int(ok)
        rows.append((gold, pred, ok, len(t_out.outputs[0].token_ids)))

    print("\n" + "=" * 72)
    print(f"BUDGET-FORCED GSM8K  (n={len(docs)}, think_budget={args.think_budget})")
    print("=" * 72)
    print(f"{'gold':>10} {'pred':>10} {'ok':>3} {'think_toks':>10}")
    for gold, pred, ok, ntok in rows[:20]:
        print(f"{gold:>10} {str(pred):>10} {'Y' if ok else '.':>3} {ntok:>10}")
    if len(rows) > 20:
        print(f"... ({len(rows) - 20} more rows in the JSON output)")
    print("-" * 72)
    acc = correct / len(docs)
    print(f"accuracy: {correct}/{len(docs)} = {acc * 100:.1f}%")
    print(f"self-closed </think> within budget: {closed}/{len(docs)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "n": len(docs),
                "think_budget": args.think_budget,
                "max_model_len": args.max_model_len,
                "temperature": args.temp,
                "top_p": args.top_p,
                "seed": 42,
                "correct": correct,
                "accuracy": acc,
                "self_closed_think": closed,
                "per_question": [{"gold": g, "pred": p, "correct": ok, "think_tokens": nt} for g, p, ok, nt in rows],
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
