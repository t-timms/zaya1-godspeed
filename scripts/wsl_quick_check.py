#!/usr/bin/env python3
"""Quick 1-prompt quality check."""
from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    from vllm import LLM, SamplingParams

    MODEL_DIR = (
        "/mnt/c/Users/ttimm/Documents/Project Portfolio/"
        "zaya1-godspeed/zaya1-8b-nvfp4-ct-gs16"
    )

    print("Loading model...")
    llm = LLM(
        model=MODEL_DIR, dtype="bfloat16", max_model_len=2048,
        trust_remote_code=True, enforce_eager=True, max_num_seqs=1,
        tokenizer="Zyphra/ZAYA1-8B",
    )

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Zyphra/ZAYA1-8B", trust_remote_code=True)

    # Try two prompts: raw text completion + chat-template
    raw_prompt = "The capital of France is"
    msgs = [{"role": "user", "content": "Explain what a binary search tree is in one sentence."}]
    chat_prompt = tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True
    )

    print(f"Raw prompt: {raw_prompt!r}")
    print(f"Chat prompt ({len(chat_prompt)} chars): {chat_prompt[:120]}...")
    print(f"Tokenizer pad_token_id={tok.pad_token_id} eos_token_id={tok.eos_token_id} bos_token_id={tok.bos_token_id}")

    sp_raw = SamplingParams(max_tokens=40, temperature=0.0)
    sp_chat = SamplingParams(max_tokens=120, temperature=0.0)

    print("\n--- RAW completion ---")
    out_raw = llm.generate([raw_prompt], sp_raw)
    for o in out_raw:
        out = o.outputs[0]
        ids = out.token_ids
        print(f"Token IDs: {list(ids)[:20]}")
        print(f"Text: {out.text!r}")
        print(f"Decoded: {tok.decode(ids)!r}")

    print("\n--- CHAT completion ---")
    out_chat = llm.generate([chat_prompt], sp_chat)
    for o in out_chat:
        out = o.outputs[0]
        ids = out.token_ids
        print(f"Token IDs (first 20): {list(ids)[:20]}")
        print(f"Text: {out.text[:200]!r}")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    sys.exit(main())
