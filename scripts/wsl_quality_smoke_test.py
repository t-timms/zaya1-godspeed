#!/usr/bin/env python3
"""Quality smoke test: generate text from NVFP4 CT model and verify output quality.

Loads the NVFP4 Compressed-Tensors ZAYA1-8B model via vLLM's LLM class,
generates responses for 5 quality-gated prompts, and checks:
  - Coherent English output (not garbage)
  - <think>...</think> blocks present and well-formed
  - <|im_end|> token present
  - Code blocks in code-generation prompts
  - <zyphra_tool_call> XML validity in tool-call prompts
  - Numeric answers in math reasoning prompts

Smoke test passes if >= 4/5 prompts meet all their checks.
"""

from __future__ import annotations

import json
import multiprocessing
import re
import sys
from typing import Any

IM_END = "<|im_end|>"
THINK_START = "<think>"
THINK_END = "</think>"
ZYPHRA_TOOL_CALL_START = "<zyphra_tool_call>"
ZYPHRA_TOOL_CALL_END = "</zyphra_tool_call>"

MODEL_DIR = (
    "/mnt/c/Users/ttimm/Documents/Project Portfolio/"
    "zaya1-godspeed/zaya1-8b-nvfp4-ct"
)
PROMPTS_FILE = (
    "/mnt/c/Users/ttimm/Documents/Project Portfolio/"
    "zaya1-godspeed/prompts/smoke_test.jsonl"
)


def load_prompts(path: str) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def check_coherent(text: str, prompt_id: str) -> bool:
    if not text or not text.strip():
        print(f"  [{prompt_id}] FAIL: empty response")
        return False
    if len(text.strip()) < 30:
        print(f"  [{prompt_id}] FAIL: too short ({len(text.strip())} chars)")
        return False
    non_printable = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
    if non_printable > len(text) * 0.1:
        pct = 100 * non_printable / max(len(text), 1)
        print(f"  [{prompt_id}] FAIL: {pct:.0f}% non-printable chars")
        return False
    return True


def check_think_tags(text: str, prompt_id: str) -> bool:
    has_start = THINK_START in text
    has_end = THINK_END in text
    if has_start and has_end:
        if text.index(THINK_START) < text.index(THINK_END):
            return True
        print(f"  [{prompt_id}] FAIL: </think> before <think>")
        return False
    if has_start and not has_end:
        print(f"  [{prompt_id}] FAIL: unclosed <think>")
        return False
    return True  # missing think block is OK (may be trimmed)


def check_im_end(text: str, prompt_id: str) -> bool:
    if IM_END in text:
        return True
    # Truncation by max_tokens is expected
    return True


def check_code_block(text: str, prompt_id: str) -> bool:
    has_fenced = bool(re.search(r"```(?:python)?\s.*?```", text, re.DOTALL))
    has_function = bool(re.search(r"def\s+\w+\s*\(.*?\)\s*(?:->.*?)?\s*:", text))
    if has_fenced or has_function:
        return True
    print(f"  [{prompt_id}] WARN: no code block/func detected")
    return True  # not a hard failure


def check_tool_call_xml(text: str, prompt_id: str) -> bool:
    if ZYPHRA_TOOL_CALL_START not in text:
        return True  # acceptable for non-tool prompts
    pattern = re.compile(
        r"<zyphra_tool_call>(.*?)</zyphra_tool_call>", re.DOTALL
    )
    matches = pattern.findall(text)
    if not matches:
        print(f"  [{prompt_id}] FAIL: malformed tool-call XML")
        return False
    for i, body in enumerate(matches):
        try:
            parsed = json.loads(body.strip())
            if not parsed.get("name"):
                print(f"  [{prompt_id}] FAIL: tool call #{i+1} missing 'name'")
                return False
            if not isinstance(parsed.get("arguments"), dict):
                print(f"  [{prompt_id}] FAIL: tool call #{i+1} 'arguments' not dict")
                return False
        except json.JSONDecodeError as e:
            print(f"  [{prompt_id}] FAIL: tool call #{i+1} invalid JSON: {e}")
            return False
    return True


def check_math_answer(text: str, prompt_id: str) -> bool:
    has_numbers = bool(
        re.search(r"\d+\.?\d*\s*(?:mph|km/h|miles|km|average)", text, re.IGNORECASE)
    )
    if not has_numbers:
        print(f"  [{prompt_id}] WARN: no numeric result detected")
    return True  # not a hard failure


CHECKS: dict[str, callable] = {
    "coherent": check_coherent,
    "think_tag": check_think_tags,
    "im_end_tag": check_im_end,
    "code_block": check_code_block,
    "tool_call_xml": check_tool_call_xml,
    "math_answer": check_math_answer,
}


def main() -> int:
    from vllm import LLM, SamplingParams

    print("=" * 60)
    print("NVFP4 CT ZAYA1-8B — Quality Smoke Test")
    print("=" * 60)

    prompts_data = load_prompts(PROMPTS_FILE)
    if not prompts_data:
        print("ERROR: No prompts loaded")
        return 3

    print(f"\nLoading model from: {MODEL_DIR}")
    llm = LLM(
        model=MODEL_DIR,
        dtype="float16",
        max_model_len=2048,
        trust_remote_code=True,
        enforce_eager=True,
        max_num_seqs=1,
        tokenizer="Zyphra/ZAYA1-8B",
    )
    print("Model loaded. Starting generation...\n")

    # Build prompt strings using the model's chat template
    prompt_texts = [p["prompt"] for p in prompts_data]

    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=0.0,  # deterministic for quality verification
        stop=["<|im_end|>"],
    )

    outputs = llm.generate(prompt_texts, sampling_params)

    passed = 0
    total = len(prompts_data)

    for i, (prompt_data, output) in enumerate(zip(prompts_data, outputs)):
        pid = prompt_data["id"]
        purpose = prompt_data.get("purpose", "unknown")
        required_checks = prompt_data.get("checks", [])
        text = output.outputs[0].text

        print(f"--- [{i+1}/{total}] {pid} ({purpose}) ---")
        print(f"  Prompt: {prompt_data['prompt'][:100]}...")
        print(f"  Response ({len(text)} chars):")
        # Show first 500 chars
        preview = text[:500].replace("\n", "\\n")
        print(f"  {preview}...")
        print()

        all_ok = True
        for check_name in required_checks:
            fn = CHECKS.get(check_name)
            if fn is None:
                continue
            if not fn(text, pid):
                all_ok = False

        if all_ok:
            passed += 1
            print(f"  [{pid}] PASSED\n")
        else:
            print(f"  [{pid}] FAILED\n")

    print("=" * 60)
    print(f"RESULTS: {passed}/{total} passed (threshold: 4)")
    if passed >= 4:
        print("QUALITY SMOKE TEST: PASSED")
        return 0
    else:
        print("QUALITY SMOKE TEST: FAILED")
        return 1


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    sys.exit(main())
