#!/usr/bin/env python3
"""Smoke test: try to load NVFP4 CT ZAYA1-8B model via vLLM.
Exits 0 if model loads successfully, non-zero on error.
"""

from __future__ import annotations

import multiprocessing
import os
import sys


def main() -> int:
    MODEL_DIR = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct-gs16"

    print(f"Loading model from: {MODEL_DIR}")
    print(f"Model exists: {os.path.isdir(MODEL_DIR)}")

    from vllm import LLM

    print("Creating LLM engine...")
    try:
        llm = LLM(
            model=MODEL_DIR,
            dtype="float16",
            max_model_len=2048,
            trust_remote_code=True,
            enforce_eager=True,
            max_num_seqs=1,
            tokenizer="Zyphra/ZAYA1-8B",
        )
        print("SUCCESS: Model loaded!")
        print(f"Model config: {llm.llm_engine.model_config}")
        return 0
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    sys.exit(main())
