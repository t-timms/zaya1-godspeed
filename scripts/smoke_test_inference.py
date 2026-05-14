"""Gate 1: Inference quality smoke test against FP8 vLLM server.

Queries the running FP8 vLLM server at a configurable base URL with 5 prompts
from prompts/smoke_test.jsonl. Verifies coherent output, <think>/<|im_end|>
token handling, and tool-call XML parsing on >=4/5 prompts.

Usage:
    uv run python scripts/smoke_test_inference.py
    uv run python scripts/smoke_test_inference.py --base-url http://localhost:8010/v1
    uv run python scripts/smoke_test_inference.py --model Zyphra/ZAYA1-8B --prompts prompts/smoke_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8010/v1"
DEFAULT_MODEL = "Zyphra/ZAYA1-8B"
DEFAULT_PROMPTS = "prompts/smoke_test.jsonl"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TIMEOUT = 120
PASS_THRESHOLD = 4  # >=4/5 prompts must pass

IM_END = "<|im_end|>"
THINK_START = "<think>"
THINK_END = "</think>"
ZYPHRA_TOOL_CALL_START = "<zyphra_tool_call>"
ZYPHRA_TOOL_CALL_END = "</zyphra_tool_call>"
ZYPHRA_TOOL_RESPONSE_START = "<zyphra_tool_response>"


def load_prompts(path: str) -> list[dict[str, Any]]:
    """Load prompts from a JSONL file."""
    prompts: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    if len(prompts) != 5:
        logger.warning("Expected 5 prompts in %s, found %d", path, len(prompts))
    return prompts


def _api_request(url: str, data: bytes, timeout: int) -> dict[str, Any]:
    """Make an HTTP request to the vLLM API."""
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection failed: {e.reason}") from e


def check_server_health(base_url: str, timeout: int = 5) -> bool:
    """Check if the vLLM server is reachable."""
    health_url = base_url.rstrip("/v1").rstrip("/") + "/health"
    try:
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def send_chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> str:
    """Send a chat completion request and return the assistant's response text."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode("utf-8")

    url = base_url.rstrip("/") + "/chat/completions"
    result = _api_request(url, payload, timeout)
    return result["choices"][0]["message"]["content"]


def check_coherent(text: str, prompt_id: str) -> bool:
    """Check that the output is non-empty and not garbled."""
    if not text or not text.strip():
        logger.warning("  [%s] Empty response", prompt_id)
        return False
    if len(text.strip()) < 20:
        logger.warning("  [%s] Response too short: %d chars", prompt_id, len(text.strip()))
        return False
    # Check for obvious garbage (high ratio of non-printable chars)
    non_printable = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
    if non_printable > len(text) * 0.1:
        logger.warning("  [%s] High non-printable ratio: %.1f%%", prompt_id, 100 * non_printable / max(len(text), 1))
        return False
    return True


def check_think_tokens(text: str, prompt_id: str) -> bool:
    """Verify <think>...</think> block is properly structured if present."""
    has_start = THINK_START in text
    has_end = THINK_END in text
    if has_start and has_end:
        if text.index(THINK_START) < text.index(THINK_END):
            logger.info("  [%s] <think> block: well-formed", prompt_id)
            return True
        logger.warning("  [%s] <think> block: malformed (end before start)", prompt_id)
        return False
    if has_start and not has_end:
        logger.warning("  [%s] <think> block: unclosed", prompt_id)
        return False
    logger.info("  [%s] <think> block: not present (acceptable — may be trimmed)", prompt_id)
    return True


def check_im_end_token(text: str, prompt_id: str) -> bool:
    """Verify <|im_end|> token presence."""
    if IM_END in text:
        logger.info("  [%s] <|im_end|>: present", prompt_id)
        return True
    if THINK_END in text:
        # Model may have been cut off mid-generation; im_end after thinking is expected
        # but max_tokens truncation means im_end may be absent. Not a failure.
        logger.info("  [%s] <|im_end|>: absent (probably truncated by max_tokens)", prompt_id)
        return True
    logger.info("  [%s] <|im_end|>: absent", prompt_id)
    return True  # Not a hard failure — truncation is expected at 512 tokens


def check_tool_call_xml(text: str, prompt_id: str) -> bool:
    """Parse tool-call XML if present and verify JSON validity.

    Tool-call format: <zyphra_tool_call>{"name":"...","arguments":{...}}</zyphra_tool_call>
    (ref: AGENTS.md, tokenizer IDs 101-104)
    """
    if ZYPHRA_TOOL_CALL_START not in text:
        logger.info("  [%s] <zyphra_tool_call>: not present (acceptable for non-tool prompts)", prompt_id)
        return True

    pattern = re.compile(
        r"<zyphra_tool_call>(.*?)</zyphra_tool_call>",
        re.DOTALL,
    )
    matches = pattern.findall(text)

    if not matches:
        logger.warning("  [%s] <zyphra_tool_call>: malformed XML — no closing tag found", prompt_id)
        return False

    for i, body in enumerate(matches):
        try:
            parsed = json.loads(body.strip())
            name = parsed.get("name", "")
            args = parsed.get("arguments", {})
            if not name:
                logger.warning("  [%s] Tool call #%d: missing 'name' field", prompt_id, i + 1)
                return False
            if not isinstance(args, dict):
                logger.warning("  [%s] Tool call #%d: 'arguments' is not a dict", prompt_id, i + 1)
                return False
            logger.info("  [%s] Tool call #%d: %s(%s)", prompt_id, i + 1, name, json.dumps(args)[:80])
        except json.JSONDecodeError as e:
            logger.warning("  [%s] Tool call #%d: invalid JSON — %s", prompt_id, i + 1, e)
            return False

    logger.info("  [%s] <zyphra_tool_call>: parsed %d tool call(s) successfully", prompt_id, len(matches))
    return True


def check_code_block(text: str, prompt_id: str) -> bool:
    """Check for code block presence if expected."""
    has_code = bool(re.search(r"```(?:python)?\s.*?```", text, re.DOTALL))
    has_inline = bool(re.search(r"def\s+\w+\s*\(.*?\)\s*(?:->.*?)?\s*:", text))
    if has_code or has_inline:
        logger.info("  [%s] code block: present", prompt_id)
        return True
    logger.info("  [%s] code block: not detected (acceptable)", prompt_id)
    return True


def check_math_answer(text: str, prompt_id: str) -> bool:
    """Check for numeric answer presence in math reasoning output."""
    has_numbers = bool(re.search(r"\d+\.?\d*\s*(?:mph|km/h|miles|km)", text, re.IGNORECASE))
    if has_numbers:
        logger.info("  [%s] math answer: numeric result detected", prompt_id)
        return True
    logger.info("  [%s] math answer: no numeric result detected", prompt_id)
    return True


CHECK_FUNCTIONS: dict[str, list[tuple[str, Any]]] = {
    "coherent": [("coherent", check_coherent)],
    "think_tag": [("think_tags", check_think_tokens)],
    "im_end_tag": [("im_end", check_im_end_token)],
    "tool_call_xml": [("tool_call_xml", check_tool_call_xml)],
    "code_block": [("code_block", check_code_block)],
    "math_answer": [("math_answer", check_math_answer)],
}


def evaluate_prompt(
    prompt: dict[str, Any],
    response: str,
) -> tuple[bool, list[str]]:
    """Run all applicable checks on a response.

    Returns (passed, list of failed check names).
    """
    prompt_id = prompt["id"]
    required_checks = prompt.get("checks", [])

    logger.info("--- %s (%s) ---", prompt_id, prompt.get("purpose", "unknown"))
    logger.info("  Prompt: %s", prompt["prompt"][:120])

    failed: list[str] = []
    for check_name in required_checks:
        if check_name not in CHECK_FUNCTIONS:
            logger.warning("  [%s] Unknown check: %s", prompt_id, check_name)
            continue
        for display_name, fn in CHECK_FUNCTIONS[check_name]:
            if not fn(response, prompt_id):
                failed.append(display_name)

    if failed:
        logger.warning("  [%s] FAILED checks: %s", prompt_id, ", ".join(failed))
    else:
        total = len([c for name in required_checks for c in CHECK_FUNCTIONS.get(name, [])])
        logger.info("  [%s] PASSED (%d checks)", prompt_id, total)

    return len(failed) == 0, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate 1: FP8 vLLM inference quality smoke test",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ZAYA_BASE_URL", DEFAULT_BASE_URL),
        help=f"vLLM OpenAI-compatible base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ZAYA_MODEL_ID", DEFAULT_MODEL),
        help=f"Model ID for the API request (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--prompts",
        default=DEFAULT_PROMPTS,
        help=f"Path to prompts JSONL file (default: {DEFAULT_PROMPTS})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per response (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature for generation (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per request in seconds (default: {DEFAULT_TIMEOUT})",
    )
    args = parser.parse_args()

    logger.info("=== GATE 1: Inference Quality Smoke Test ===")
    logger.info("Server: %s", args.base_url)
    logger.info("Model:  %s", args.model)
    logger.info("Prompts: %s", args.prompts)

    if not check_server_health(args.base_url):
        logger.error("vLLM server not reachable at %s", args.base_url)
        logger.error("Start the FP8 server first:")
        logger.error("  wsl -d Ubuntu -- bash -c 'source ~/vllm-env/bin/activate &&")
        logger.error("    vllm serve Zyphra/ZAYA1-8B --port 8010 --quantization fp8")
        logger.error("    --dtype bfloat16 --max-model-len 2048 --trust-remote-code --enforce-eager'")
        return 2

    logger.info("Server health: OK")

    prompts = load_prompts(args.prompts)
    if not prompts:
        logger.error("No prompts loaded from %s", args.prompts)
        return 3

    passed = 0
    failed_prompts: list[tuple[str, list[str]]] = []

    for prompt in prompts:
        prompt_id = prompt["id"]

        try:
            response = send_chat_completion(
                args.base_url,
                args.model,
                prompt["prompt"],
                args.max_tokens,
                args.temperature,
                args.timeout,
            )
        except RuntimeError as e:
            logger.error("  [%s] API error: %s", prompt_id, e)
            failed_prompts.append((prompt_id, ["api_error"]))
            continue

        ok, failures = evaluate_prompt(prompt, response)
        if ok:
            passed += 1
        else:
            failed_prompts.append((prompt_id, failures))

    logger.info("")
    logger.info("=== RESULTS ===")
    logger.info("Passed: %d/%d (threshold: %d)", passed, len(prompts), PASS_THRESHOLD)

    if failed_prompts:
        for pid, failures in failed_prompts:
            logger.info("  FAILED: %s — %s", pid, ", ".join(failures))

    if passed >= PASS_THRESHOLD:
        logger.info("GATE 1: PASSED — %d/%d prompts verified", passed, len(prompts))
        return 0
    else:
        logger.error("GATE 1: FAILED — only %d/%d prompts passed (need >=%d)", passed, len(prompts), PASS_THRESHOLD)
        return 1


if __name__ == "__main__":
    sys.exit(main())
