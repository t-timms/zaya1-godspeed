"""Diagnostic: dump ZAYA1-8B's raw generations on a few GPQA-Diamond questions.

The standard lm-eval `gpqa_diamond_cot_zeroshot` task scored 0.0% strict-match
(needs the literal "The answer is X") and 6.6% flexible-extract on our W4A4
checkpoint — below random. ZAYA1-8B is a reasoning model with its own <think>
answer format, so the stock extraction never matches. This script prints the
model's actual output so we can see (a) whether it reasons coherently and
(b) what answer format it really uses, to build correct extraction.

Run (WSL, vllm-env active):
    python3 scripts/diagnose_gpqa_format.py
"""

from __future__ import annotations

import os
import random

os.environ.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")

MODEL = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-w4a4"
N_QUESTIONS = 4
MAX_TOKENS = int(os.environ.get("DIAG_MAX_TOKENS", "4096"))
TEMPERATURE = float(os.environ.get("DIAG_TEMP", "0.0"))
TOP_P = float(os.environ.get("DIAG_TOP_P", "1.0"))


def _preprocess(text: str | None) -> str:
    if text is None:
        return " "
    return text.strip().replace(" [title]", ". ").replace("  ", " ")


def build_docs(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    random.seed(42)
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
        docs.append(
            {
                "question": _preprocess(row["Question"]),
                "choices": choices,
                "answer": f"({chr(65 + idx)})",
            }
        )
    return docs


def main() -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = build_docs(N_QUESTIONS)

    prompts: list[str] = []
    for d in docs:
        user_text = (
            f"What is the correct answer to this question:{d['question']}\n"
            f"Choices:\n(A) {d['choices'][0]}\n(B) {d['choices'][1]}\n"
            f"(C) {d['choices'][2]}\n(D) {d['choices'][3]}\n"
            "Let's think step by step: "
        )
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    print("=" * 88)
    print("FIRST RENDERED PROMPT (chat-templated, what the model actually sees):")
    print("=" * 88)
    print(prompts[0])
    print("=" * 88)

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        moe_backend="cutlass",
        gpu_memory_utilization=0.92,
        max_model_len=8192,
        enforce_eager=True,
        enable_prefix_caching=False,
        kv_cache_dtype="fp8",
    )
    sp = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        stop=["</s>"],
        seed=42,
    )
    print(f"\n[sampling] temp={TEMPERATURE} top_p={TOP_P} max_tokens={MAX_TOKENS}\n")
    outs = llm.generate(prompts, sp)

    for i, (d, out) in enumerate(zip(docs, outs)):
        text = out.outputs[0].text
        print("\n" + "#" * 88)
        print(f"Q{i + 1}  gold={d['answer']}  generated_tokens={len(out.outputs[0].token_ids)}")
        print("#" * 88)
        print(text)
        print("-" * 88)
        print(f"[last 300 chars] ...{text[-300:]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
