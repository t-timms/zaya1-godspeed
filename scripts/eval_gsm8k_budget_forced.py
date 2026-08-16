"""Budget-forced GSM8K for ZAYA1-8B W4A4 on 16 GB VRAM.

Why not the standard tool: lm-eval-harness *does* support reasoning models via
`think_end_token`, but it only strips post-hoc — `generation.split(tok)[-1]` —
so when the model never emits `</think>` within the budget it returns the whole
unterminated trace as the "answer" and scores that (the same artifact that put
IFEval at 19.8% vs a BF16 reference of 85.58%). ZAYA1 frequently does not close
its think block, so this implements actual budget *forcing* instead:
  Stage 1  generate up to --think-budget reasoning tokens (stop early on </think>).
  Stage 2  inject "</think>\n\nThe answer is " and decode a short numeric answer.
  Score    reuse lm-eval's own gsm8k-cot-zeroshot extraction regexes so the
           number-parsing logic matches the standard harness exactly.

Sampling defaults follow Zyphra's published recommendation for ZAYA1-8B
(temperature 0.6, top_p 0.95 for agent/code tasks; they recommend 1.0 for
general use). Note their published benchmark numbers come from a private
"Zyphra evaluation harness" with undisclosed generation limits, so cross-
comparison against them is indicative, not a reproduction.

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


def wilson_ci(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    A bare accuracy number is not interpretable on its own — 75% vs 80% on
    n=20 looks like a difference and is not (that pair tested at p=1.0000).
    Wilson rather than the normal approximation because it stays sensible at
    small n and near 0/1, which subset runs hit.
    """
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z / denom * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return (max(0.0, center - half), min(1.0, center + half))


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
    # Exposed so a multi-seed variance check is possible. The Wilson interval
    # reported below covers item-sampling uncertainty only; it says nothing
    # about generation stochasticity at temperature > 0. Quantifying that needs
    # repeat runs at different seeds (this repo's own rule: n=1 is not a
    # result), which is why the knob exists even though the default is fixed.
    ap.add_argument("--seed", type=int, default=42)
    # ZAYA1's chat template ships an enable_thinking flag (Zyphra's own). Off,
    # it pre-closes <think> so the model answers directly — measured 3x faster
    # with 5x fewer tokens. This runs that configuration instead of budget
    # forcing, to quantify what disabling reasoning costs in accuracy.
    ap.add_argument("--no-thinking", action="store_true", dest="no_thinking")
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

    if args.no_thinking:
        # ZAYA1's chat template pre-closes <think> when enable_thinking=False,
        # so the model answers immediately and no budget forcing is needed.
        # Routed through llm.chat() rather than apply_chat_template()+generate():
        # that path has twice produced fluent-but-off-topic output on this model
        # (doubled BOS — the template emits bos_token itself).
        # NOTE this makes the comparison protocol-vs-protocol (two-stage forced
        # vs single-stage direct), not a single-variable ablation. Label it so.
        convs = [[{"role": "user", "content": d["question"]}] for d in docs]
        sp = SamplingParams(temperature=args.temp, top_p=args.top_p, max_tokens=1024, seed=args.seed)
        ans_outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False})
        think_outs = ans_outs  # no separate trace; keeps downstream zip() uniform
        closed = 0
    else:
        sp_think = SamplingParams(
            temperature=args.temp,
            top_p=args.top_p,
            max_tokens=args.think_budget,
            stop=["</think>", "</s>"],
            seed=args.seed,
        )
        think_outs = llm.generate(base_prompts, sp_think)
        closed = sum(o.outputs[0].stop_reason == "</think>" for o in think_outs)

        forced_prompts = [
            base + out.outputs[0].text + "\n</think>\n\nThe answer is " for base, out in zip(base_prompts, think_outs)
        ]
        sp_ans = SamplingParams(temperature=0.0, max_tokens=32, stop=["\n", "</s>"], seed=42)
        ans_outs = llm.generate(forced_prompts, sp_ans)

    correct = 0
    extraction_failures = 0
    rows = []
    for d, t_out, a_out in zip(docs, think_outs, ans_outs):
        gold = gold_answer(d["answer"])
        pred = extract_number(a_out.outputs[0].text)
        # Fall back to the tail of the reasoning trace when the forced short
        # answer yields no number at all (the model opened with prose or a
        # newline and hit the stop sequence). Without this, a formatting miss
        # is scored as a wrong answer and silently understates accuracy — it
        # cost 1 of 20 items on the first smoke run. Mirrors the same fallback
        # already used in eval_gpqa_budget_forced.py / eval_mmlu_pro_*.py.
        if pred is None:
            pred = extract_number(t_out.outputs[0].text[-200:])
            if pred is not None:
                extraction_failures += 1
        ok = pred is not None and pred == gold
        correct += int(ok)
        rows.append((gold, pred, ok, len(t_out.outputs[0].token_ids)))

    print("\n" + "=" * 72)
    mode = (
        "NO-THINKING (enable_thinking=False)" if args.no_thinking else f"BUDGET-FORCED think_budget={args.think_budget}"
    )
    print(f"GSM8K  (n={len(docs)}, {mode})")
    print("=" * 72)
    print(f"{'gold':>10} {'pred':>10} {'ok':>3} {'think_toks':>10}")
    for gold, pred, ok, ntok in rows[:20]:
        print(f"{gold:>10} {str(pred):>10} {'Y' if ok else '.':>3} {ntok:>10}")
    if len(rows) > 20:
        print(f"... ({len(rows) - 20} more rows in the JSON output)")
    print("-" * 72)
    acc = correct / len(docs)
    ci_lo, ci_hi = wilson_ci(correct, len(docs))
    print(f"accuracy: {correct}/{len(docs)} = {acc * 100:.1f}%  95% CI [{ci_lo * 100:.1f}, {ci_hi * 100:.1f}]")
    print(f"self-closed </think> within budget: {closed}/{len(docs)}")
    print(f"recovered by trace fallback (forced answer had no number): {extraction_failures}/{len(docs)}")
    truncated = sum(1 for _, _, _, nt in rows if nt >= args.think_budget)
    print(f"hit think-budget ceiling: {truncated}/{len(docs)}  (raise --think-budget if this is most of them)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "n": len(docs),
                "think_budget": None if args.no_thinking else args.think_budget,
                "no_thinking": args.no_thinking,
                "max_model_len": args.max_model_len,
                "temperature": args.temp,
                "top_p": args.top_p,
                "seed": args.seed,
                "correct": correct,
                "accuracy": acc,
                "accuracy_ci95": [ci_lo, ci_hi],
                "self_closed_think": closed,
                "recovered_by_trace_fallback": extraction_failures,
                "hit_think_budget_ceiling": truncated,
                "per_question": [{"gold": g, "pred": p, "correct": ok, "think_tokens": nt} for g, p, ok, nt in rows],
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
