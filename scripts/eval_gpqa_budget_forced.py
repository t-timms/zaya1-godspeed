"""Budget-forced GPQA-Diamond eval for ZAYA1-8B W4A4 on 16 GB VRAM.

Why this exists: ZAYA1-8B is a long-reasoning model. Its <think> trace does not
terminate within a feasible token budget on a 16 GB GPU (diagnosed 2026-06-22:
traces hit 7000 tokens without ever emitting </think>, some degenerate into
repetition). The stock lm-eval gpqa_diamond_cot_zeroshot task therefore scores it
near zero — a harness/budget artifact, not quantization damage (the reasoning is
coherent and reaches correct answers when read by hand).

This implements "budget forcing" (s1-style): generate up to THINK_BUDGET reasoning
tokens, then inject "</think>\n\nThe final answer is (" and decode a few more tokens
to extract the choice letter. This is both a fair, terminating eval protocol and the
way the model would actually be served under a fixed latency/VRAM budget.

Run (WSL, vllm-env active):
    python3 scripts/eval_gpqa_budget_forced.py --n 40 --think-budget 2500
"""

from __future__ import annotations

import argparse
import os
import random
import re

os.environ.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")

MODEL = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-w4a4"


def _preprocess(text: str | None) -> str:
    if text is None:
        return " "
    return text.strip().replace(" [title]", ". ").replace("  ", " ")


def build_docs(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    random.seed(42)
    n = min(n, len(ds))
    docs = []
    for row in ds.select(range(n)):
        choices = [
            _preprocess(row["Incorrect Answer 1"]),
            _preprocess(row["Incorrect Answer 2"]),
            _preprocess(row["Incorrect Answer 3"]),
            _preprocess(row["Correct Answer"]),
        ]
        random.shuffle(choices)
        idx = choices.index(_preprocess(row["Correct Answer"]))
        docs.append({"choices": choices, "gold": chr(65 + idx), "question": _preprocess(row["Question"])})
    return docs


# Accept "(C)", "C", "answer is C", "\boxed{C}" in the short continuation.
_LETTER_RE = re.compile(r"\(?\s*([ABCD])\s*\)?")


def extract_letter(continuation: str) -> str | None:
    m = re.search(r"boxed\{\s*([ABCD])", continuation)
    if m:
        return m.group(1)
    m = _LETTER_RE.search(continuation)
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of GPQA-Diamond questions")
    ap.add_argument("--think-budget", type=int, default=2500, dest="think_budget")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    ap.add_argument("--max-model-len", type=int, default=8192, dest="max_model_len")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = build_docs(args.n)

    base_prompts: list[str] = []
    for d in docs:
        user_text = (
            f"What is the correct answer to this question:{d['question']}\n"
            f"Choices:\n(A) {d['choices'][0]}\n(B) {d['choices'][1]}\n"
            f"(C) {d['choices'][2]}\n(D) {d['choices'][3]}\n"
            "Let's think step by step: "
        )
        base_prompts.append(
            tok.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

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

    # Stage 1: bounded reasoning. Stop early if the model closes </think> on its own.
    sp_think = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.think_budget,
        stop=["</think>", "</s>"],
        seed=42,
    )
    think_outs = llm.generate(base_prompts, sp_think)

    # Stage 2: force the conclusion and read the letter.
    forced_prompts: list[str] = []
    closed_count = 0
    for base, out in zip(base_prompts, think_outs):
        trace = out.outputs[0].text
        stop_reason = out.outputs[0].stop_reason
        if stop_reason == "</think>":
            closed_count += 1
        forced_prompts.append(base + trace + "\n</think>\n\nThe final answer is (")

    sp_ans = SamplingParams(temperature=0.0, max_tokens=8, stop=[")", "\n"], seed=42)
    ans_outs = llm.generate(forced_prompts, sp_ans)

    correct = 0
    rows = []
    for d, t_out, a_out in zip(docs, think_outs, ans_outs):
        cont = a_out.outputs[0].text
        pred = extract_letter(cont) or extract_letter(t_out.outputs[0].text[-200:])
        ok = pred == d["gold"]
        correct += int(ok)
        rows.append((d["gold"], pred, ok, len(t_out.outputs[0].token_ids), repr(cont[:20])))

    print("\n" + "=" * 72)
    print(f"BUDGET-FORCED GPQA-Diamond  (n={len(docs)}, think_budget={args.think_budget})")
    print("=" * 72)
    print(f"{'gold':>4} {'pred':>4} {'ok':>3} {'think_toks':>10}  continuation")
    for gold, pred, ok, ntok, cont in rows:
        print(f"{gold:>4} {str(pred):>4} {'Y' if ok else '.':>3} {ntok:>10}  {cont}")
    print("-" * 72)
    acc = correct / len(docs)
    print(f"accuracy: {correct}/{len(docs)} = {acc * 100:.1f}%")
    print(f"self-closed </think> within budget: {closed_count}/{len(docs)}")
    print(f"BF16 reference (Zyphra CoT): 71.0%   random: 25.0%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
