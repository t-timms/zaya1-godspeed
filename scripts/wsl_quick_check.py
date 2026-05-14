#!/usr/bin/env python3
"""Quick 1-prompt quality check."""
from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    from vllm import LLM, SamplingParams

    MODEL_DIR = (
        "/mnt/c/Users/ttimm/Documents/Project Portfolio/"
        "zaya1-godspeed/zaya1-8b-nvfp4-ct"
    )

    print("Loading model...")
    llm = LLM(
        model=MODEL_DIR, dtype="float16", max_model_len=2048,
        trust_remote_code=True, enforce_eager=True, max_num_seqs=1,
        tokenizer="Zyphra/ZAYA1-8B",
    )

    # Zaya uses Qwen-style chat template with <think> blocks
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Zyphra/ZAYA1-8B", trust_remote_code=True)
    msgs = [{"role": "user", "content": "Explain what a binary search tree is in one sentence."}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                      enable_thinking=True)

    print(f"Prompt ({len(prompt)} chars): {prompt[:120]}...")

    sp = SamplingParams(max_tokens=120, temperature=0.0)

    print("Generating...")
    outputs = llm.generate([prompt], sp)
    for o in outputs:
        out = o.outputs[0]
        text = out.text
        token_ids = out.token_ids
        print(f"\nToken IDs generated: {len(token_ids)}")
        if token_ids:
            print(f"First 10 IDs: {token_ids[:10]}")
            print(f"Last 10 IDs: {token_ids[-10:]}")
        print(f"Text ({len(text)} chars): {text[:300]!r}")
        print()
        if len(token_ids) > 0 and len(text) == 0:
            print("WARNING: tokens generated but text is empty — decoding issue")
            # Try decoding with the tokenizer
            full_text = tok.decode(token_ids[:20])
            print(f"Tokenizer decode first 20 tokens: {full_text!r}")
            full_text_all = tok.decode(token_ids)
            print(f"Tokenizer decode all: {full_text_all[:200]!r}")
    print("\nDONE")
    return 0


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    sys.exit(main())
