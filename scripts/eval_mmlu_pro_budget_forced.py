"""Budget-forced MMLU-Pro for ZAYA1-8B W4A4 on 16 GB VRAM.

Same protocol as scripts/eval_gpqa_budget_forced.py, extended to MMLU-Pro's
10-way multiple choice across 14 subjects (12,032 test items total):
  Stage 1  generate up to --think-budget reasoning tokens (stop early on </think>).
  Stage 2  inject "</think>\n\nThe answer is (" and decode a few tokens to read
           the letter.
  Score    reuse lm-eval's own mmlu_pro extraction regex (`answer is \\(?([A-J])\\)?`)
           for consistency with the standard harness.

The full benchmark is large - budget accordingly (a few hours at this
project's measured single-stream throughput; see RESEARCH.md). Default is a
stratified sample across all 14 subjects, not the full set - pass --n -1 for
the complete 12,032-item run and expect it to take a while. Sampling is
seeded (42) and reproducible.

Run (WSL, vllm-env active):
    python3 scripts/eval_mmlu_pro_budget_forced.py --n 20          # smoke
    python3 scripts/eval_mmlu_pro_budget_forced.py --n 700         # stratified subset
    python3 scripts/eval_mmlu_pro_budget_forced.py --n -1          # full 12,032
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

os.environ.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")

DEFAULT_MODEL = "./zaya1-8b-nvfp4-w4a4-arcbase"
CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# lm-eval's own mmlu_pro extraction regex - reused verbatim for consistency.
_ANSWER_RE = re.compile(r"answer is \(?([ABCDEFGHIJ])\)?")


def extract_letter(text: str) -> str | None:
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1)
    # Fall back to a bare letter if the model skipped the "answer is" phrasing
    m = re.search(r"\(?\s*([A-J])\s*\)?", text)
    return m.group(1) if m else None


def build_docs(n: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    docs = list(ds)
    if n is None or n < 0:
        return docs

    # Stratified sample: proportional draw from each subject category so a
    # partial run isn't accidentally skewed toward whichever subjects sort
    # first in the dataset.
    random.seed(42)
    by_subject: dict[str, list[dict]] = {}
    for d in docs:
        by_subject.setdefault(d["category"], []).append(d)
    subjects = sorted(by_subject)
    per_subject = max(1, n // len(subjects))
    sample = []
    for s in subjects:
        pool = by_subject[s]
        random.shuffle(pool)
        sample.extend(pool[:per_subject])
    random.shuffle(sample)
    return sample[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint path")
    ap.add_argument("--n", type=int, default=700, help="stratified sample size; -1 for the full 12,032-item set")
    ap.add_argument("--think-budget", type=int, default=2048, dest="think_budget")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    ap.add_argument("--max-model-len", type=int, default=8192, dest="max_model_len")
    ap.add_argument("--output", default="results/mmlu_pro_budget_forced.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    docs = build_docs(args.n)

    base_prompts = []
    for d in docs:
        lines = [f"Question:\n{d['question']}\nOptions:"]
        for i, opt in enumerate(d["options"]):
            lines.append(f"{CHOICES[i]}. {opt.strip()}")
        lines.append("Let's think step by step.")
        base_prompts.append(
            tok.apply_chat_template(
                [{"role": "user", "content": "\n".join(lines)}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

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
        base + out.outputs[0].text + "\n</think>\n\nThe answer is (" for base, out in zip(base_prompts, think_outs)
    ]
    sp_ans = SamplingParams(temperature=0.0, max_tokens=8, stop=[")", "\n"], seed=42)
    ans_outs = llm.generate(forced_prompts, sp_ans)

    correct = 0
    by_subject_correct: dict[str, list[int]] = {}
    rows = []
    for d, t_out, a_out in zip(docs, think_outs, ans_outs):
        pred = extract_letter(a_out.outputs[0].text) or extract_letter(t_out.outputs[0].text[-200:])
        ok = pred == d["answer"]
        correct += int(ok)
        by_subject_correct.setdefault(d["category"], []).append(int(ok))
        rows.append((d["category"], d["answer"], pred, ok, len(t_out.outputs[0].token_ids)))

    print("\n" + "=" * 72)
    print(f"BUDGET-FORCED MMLU-Pro  (n={len(docs)}, think_budget={args.think_budget})")
    print("=" * 72)
    print(f"{'subject':<20} {'n':>5} {'acc':>7}")
    for subj in sorted(by_subject_correct):
        vals = by_subject_correct[subj]
        print(f"{subj:<20} {len(vals):>5} {sum(vals) / len(vals) * 100:>6.1f}%")
    print("-" * 72)
    acc = correct / len(docs)
    print(f"overall accuracy: {correct}/{len(docs)} = {acc * 100:.1f}%")
    print(f"self-closed </think> within budget: {closed}/{len(docs)}")
    print("random baseline: 10.0%   (10-way multiple choice)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "n": len(docs),
                "full_set": args.n is not None and args.n < 0,
                "think_budget": args.think_budget,
                "max_model_len": args.max_model_len,
                "temperature": args.temp,
                "top_p": args.top_p,
                "seed": 42,
                "correct": correct,
                "accuracy": acc,
                "self_closed_think": closed,
                "by_subject": {s: {"n": len(v), "accuracy": sum(v) / len(v)} for s, v in by_subject_correct.items()},
                "per_question": [
                    {"category": c, "gold": g, "pred": p, "correct": ok, "think_tokens": nt} for c, g, p, ok, nt in rows
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
