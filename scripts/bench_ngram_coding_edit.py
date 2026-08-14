"""Compares baseline vs. n-gram speculative decoding on a realistic code-edit
prompt. vllm bench latency only generates synthetic random tokens, so it
can't test this — n-gram decoding's speedup depends entirely on real
prompt/output token overlap, which a synthetic benchmark has none of.

Usage (run from the repo root, 5x each for the reps this project's numbers
are based on):
    python scripts/bench_ngram_coding_edit.py baseline
    python scripts/bench_ngram_coding_edit.py ngram
"""

import sys
import time

from vllm import LLM, SamplingParams

USE_NGRAM = sys.argv[1] == "ngram" if len(sys.argv) > 1 else False

with open("scripts/verify_w4a4_dequant.py", encoding="utf-8") as f:
    source = f.read()

PROMPT = (
    "Here is a Python script. Add a one-line docstring to the top of the file "
    "describing what it does, and add a `from __future__ import annotations` "
    "import at the very top. Return the complete file with those two changes, "
    "unchanged otherwise. Output only the code, no explanation.\n\n"
    f"```python\n{source}\n```"
)


def main() -> None:
    kwargs: dict = dict(
        model="./zaya1-8b-nvfp4-w4a4-arcbase",
        dtype="bfloat16",
        kv_cache_dtype="fp8",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enforce_eager=True,
    )
    if USE_NGRAM:
        kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 2,
        }

    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=500)

    conversation = [{"role": "user", "content": PROMPT}]
    t0 = time.perf_counter()
    outputs = llm.chat([conversation], params)
    elapsed = time.perf_counter() - t0

    out = outputs[0].outputs[0]
    n_tokens = len(out.token_ids)
    tok_s = n_tokens / elapsed

    print("=== RESULT ===")
    print(f"mode={'ngram' if USE_NGRAM else 'baseline'}")
    print(f"elapsed_s={elapsed:.3f}")
    print(f"n_tokens={n_tokens}")
    print(f"tok_s={tok_s:.3f}")
    print("=== OUTPUT (last 300 chars) ===")
    print(out.text[-300:])


if __name__ == "__main__":
    main()
