"""Budget-forced HumanEval for ZAYA1-8B W4A4 on 16 GB VRAM.

Why not the standard tool: lm-eval-harness supports reasoning models via
`think_end_token`, but it only strips post-hoc — `generation.split(tok)[-1]` —
so a model that never emits `</think>` within budget has its whole
unterminated trace scored as the answer (here: submitted as the code). ZAYA1
frequently does not close its think block, so this implements actual budget
*forcing* instead.

No BF16 reference exists for this benchmark: Zyphra does not publish a
HumanEval number for ZAYA1-8B, so these results stand alone rather than as a
retention-vs-baseline comparison.

Same protocol as scripts/eval_ifeval_budget_forced.py (full-response style,
not a short forced answer - there's no short answer to force for code gen):
  Stage 1  generate up to --think-budget reasoning tokens (stop early on </think>).
  Stage 2  inject "</think>\n\n" and generate the actual code completion.
  Score    reuse HF `evaluate`'s official "code_eval" metric (the same sandboxed
           execution lm-eval's humaneval task itself uses) for pass@1 - this
           actually RUNS the generated code against the hidden test cases.

Executing model-generated code is inherent to HumanEval as a benchmark, not
something introduced by this script - HF gates this behind an explicit env
var, which is set below.

Run (WSL, vllm-env active):
    python3 scripts/eval_humaneval_budget_forced.py --n 10       # smoke
    python3 scripts/eval_humaneval_budget_forced.py              # full 164
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

DEFAULT_MODEL = "./zaya1-8b-nvfp4-w4a4-arcbase"


def extract_code(response: str, doc_prompt: str) -> str:
    """Mirror lm-eval's humaneval_instruct build_predictions_instruct: take the
    fenced code block if present, otherwise the raw text, and prepend the
    function signature the model was given so the completion is runnable."""
    text = response
    if "```python" in text:
        text = text.split("```python", 1)[1]
    elif "```" in text:
        text = text.split("```", 1)[1]
    if "```" in text:
        text = text.split("```", 1)[0]
    text = text.strip()
    # If the model already re-emitted the signature, don't double it up.
    if text.startswith(doc_prompt.strip().split("\n")[0][:20]):
        return text
    return doc_prompt + "\n" + text if not text.startswith(doc_prompt) else text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint path")
    ap.add_argument("--n", type=int, default=None, help="limit to first N problems (default: all 164)")
    ap.add_argument("--think-budget", type=int, default=2048, dest="think_budget")
    ap.add_argument("--response-max", type=int, default=1024, dest="response_max")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    ap.add_argument("--max-model-len", type=int, default=8192, dest="max_model_len")
    ap.add_argument("--output", default="results/humaneval_budget_forced.json")
    args = ap.parse_args()

    import evaluate as hf_evaluate
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    code_eval = hf_evaluate.load("code_eval")

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset("openai/openai_humaneval", split="test")
    if args.n is not None:
        ds = ds.select(range(min(args.n, len(ds))))
    docs = list(ds)

    base_prompts = [
        tok.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": (
                        "Write a solution to the following problem and make sure "
                        f"that it passes the tests:\n```python\n{d['prompt']}\n```"
                    ),
                }
            ],
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

    forced_prompts = [base + out.outputs[0].text + "\n</think>\n\n" for base, out in zip(base_prompts, think_outs)]
    sp_ans = SamplingParams(
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.response_max,
        stop=["</s>", "<|im_end|>"],
        seed=42,
    )
    ans_outs = llm.generate(forced_prompts, sp_ans)

    references = [d["test"] + f"\ncheck({d['entry_point']})" for d in docs]
    predictions = [[extract_code(a_out.outputs[0].text.strip(), d["prompt"])] for d, a_out in zip(docs, ans_outs)]

    pass_at_1, detail = code_eval.compute(references=references, predictions=predictions, k=[1])

    correct = sum(1 for d in detail.values() if d[0][1]["passed"])
    per_task = [{"task_id": d["task_id"], "passed": bool(detail[i][0][1]["passed"])} for i, d in enumerate(docs)]

    print("\n" + "=" * 72)
    print(f"BUDGET-FORCED HumanEval  (n={len(docs)}, think_budget={args.think_budget})")
    print("=" * 72)
    print(f"pass@1: {pass_at_1['pass@1'] * 100:.1f}%  ({correct}/{len(docs)})")
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
                "pass_at_1": pass_at_1["pass@1"],
                "correct": correct,
                "self_closed_think": closed,
                "per_task": per_task,
            },
            indent=2,
        )
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
