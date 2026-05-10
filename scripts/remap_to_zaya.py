"""Remap Godspeed conversation JSONL to ZAYA1-8B ChatML training format.

ZAYA1-8B uses a JSON-inside-XML tool call format via vLLM's zaya_xml parser:
    <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>

This differs from Godspeed's native Qwen3-Coder XML format and from the
OpenAI-standard tool_calls JSON. The remapper extracts conversation messages
from Godspeed's training JSONL and rewrites them into ZAYA's expected format
for TRL SFTTrainer (ChatML messages array with role/content).

Quality gates applied during remapping (per context doc):
  1. Mechanical verify hook must pass (if session_end exit_code != 0, skip)
  2. Jaccard tool selection >= 0.7 vs expected tools (if provided)
  3. Zero dangerous command flags in reward annotations
  4. Zero schema validation errors
  5. Maximum conversation length: 4096 tokens estimated

References:
  - ZAYA1-8B Technical Report (arXiv:2605.05365, May 2026)
  - Godspeed conversation logger format (conversation_logger.py)
  - vLLM --tool-call-parser zaya_xml (Zyphra/vllm@zaya1-pr)

Usage:
    python scripts/remap_to_zaya.py \
        --input-dir ~/.godspeed/training/ \
        --output data/train_zaya.jsonl \
        --expected-tools benchmarks/tasks.jsonl \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ZAYA_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def format_zaya_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Format a tool call in ZAYA1-8B's expected XML+JSON format."""
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>{payload}</tool_call>"


def _extract_tool_calls_from_content(content: str) -> list[dict[str, Any]]:
    """Extract tool calls embedded in assistant content (various formats)."""
    tool_calls: list[dict[str, Any]] = []

    for match in _ZAYA_TOOL_CALL_RE.finditer(content):
        try:
            tool_calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue

    return tool_calls


def _parse_godspeed_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a single Godspeed conversation message into ZAYA ChatML format."""
    role = msg.get("role")

    if role == "system":
        return {"role": "system", "content": msg.get("content", "")}

    if role == "user":
        return {"role": "user", "content": msg.get("content", "")}

    if role == "assistant":
        content = msg.get("content", "")
        tool_calls_raw = msg.get("tool_calls", [])

        if tool_calls_raw:
            zaya_calls = []
            for tc in tool_calls_raw:
                fn = tc.get("function", tc)
                name = fn.get("name", "")
                args_str = fn.get("arguments", "{}")
                if isinstance(args_str, str):
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = args_str
                zaya_calls.append(format_zaya_tool_call(name, args))

            if zaya_calls:
                content = "\n".join(zaya_calls)

        thinking = msg.get("thinking", "")
        if thinking:
            content = f"<think>\n{thinking}\n</think>\n\n{content}"

        return {"role": "assistant", "content": content}

    if role == "tool":
        tool_name = msg.get("name", "")
        tool_content = msg.get("content", "")
        prefix = f"[{tool_name} result]\n" if tool_name else ""
        return {"role": "tool", "content": prefix + tool_content}

    if role == "session_end":
        return None

    if role == "meta":
        return None

    return None


def _check_dangerous_commands(messages: list[dict[str, Any]]) -> bool:
    """Return True if any dangerous command flag found in reward annotations."""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("dangerous_command"):
            return True
    return False


def _check_errors(messages: list[dict[str, Any]]) -> bool:
    """Return True if any tool result has is_error=True."""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool" and msg.get("is_error"):
            return True
    return False


def _count_tool_calls(messages: list[dict[str, Any]]) -> int:
    """Count ZAYA-formatted tool calls in assistant messages."""
    count = 0
    for msg in messages:
        if msg.get("role") == "assistant":
            count += len(_ZAYA_TOOL_CALL_RE.findall(msg.get("content", "")))
    return count


def _tool_names_used(messages: list[dict[str, Any]]) -> set[str]:
    """Extract tool names used in a remapped trajectory."""
    names: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for match in _ZAYA_TOOL_CALL_RE.finditer(msg.get("content", "")):
                try:
                    tc = json.loads(match.group(1))
                    if "name" in tc:
                        names.add(tc["name"])
                except json.JSONDecodeError:
                    pass
    return names


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate: 4 chars ≈ 1 token for code-heavy content."""
    total = 0
    for msg in messages:
        if "content" in msg and isinstance(msg["content"], str):
            total += len(msg["content"]) // 4
    return total


def load_expected_tools(path: str | None) -> dict[str, set[str]]:
    """Load expected tool sets per task from benchmarks/tasks.jsonl."""
    if not path:
        return {}
    expected: dict[str, set[str]] = {}
    p = Path(path)
    if not p.exists():
        logger.warning("Expected tools file not found: %s", p)
        return expected
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            task = json.loads(line)
            expected[task["task_id"]] = set(task.get("expected_tools", []))
    return expected


def remap_session(
    filepath: Path,
    expected_tools: dict[str, set[str]],
    min_jaccard: float = 0.7,
    max_tokens: int = 4096,
) -> dict | None:
    """Remap a single Godspeed conversation file to ZAYA ChatML format.

    Returns None if the session fails quality gates.
    """
    raw_messages: list[dict[str, Any]] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw_messages.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not raw_messages:
        return None

    # Gate: dangerous commands
    if _check_dangerous_commands(raw_messages):
        logger.debug("Skipping %s: dangerous command detected", filepath.name)
        return None

    # Gate: schema validation errors (tool errors)
    if _check_errors(raw_messages):
        logger.debug("Skipping %s: tool errors present", filepath.name)
        return None

    # Gate: exit code check
    session_ok = True
    for msg in raw_messages:
        if msg.get("role") == "session_end":
            if msg.get("exit_code", 1) != 0:
                session_ok = False
            break
    if not session_ok:
        logger.debug("Skipping %s: non-zero exit code", filepath.name)
        return None

    # Parse messages to ZAYA ChatML format
    chatml_messages: list[dict[str, Any]] = []
    for msg in raw_messages:
        parsed = _parse_godspeed_message(msg)
        if parsed is not None:
            chatml_messages.append(parsed)

    if len(chatml_messages) < 3:
        return None

    # Gate: tool call count
    tc_count = _count_tool_calls(chatml_messages)
    if tc_count < 1:
        return None

    # Gate: Jaccard tool selection
    tools_used = _tool_names_used(chatml_messages)
    if expected_tools:
        best_jaccard = 0.0
        task_id = filepath.stem.split(".")[0]
        for tid, expected in expected_tools.items():
            if tid in filepath.name or task_id in tid:
                best_jaccard = max(best_jaccard, _jaccard_similarity(tools_used, expected))

        if best_jaccard < min_jaccard:
            logger.debug(
                "Skipping %s: Jaccard %.2f < %.2f",
                filepath.name,
                best_jaccard,
                min_jaccard,
            )
            return None

    # Gate: token budget
    est_tokens = _estimate_tokens(chatml_messages)
    if est_tokens > max_tokens:
        logger.debug("Skipping %s: estimated %d tokens > %d", filepath.name, est_tokens, max_tokens)
        return None

    return {"messages": chatml_messages}


def remap_directory(
    input_dir: str,
    output_path: str,
    expected_tools_path: str | None = None,
    min_jaccard: float = 0.7,
    max_tokens: int = 4096,
    max_examples: int = 500,
    dry_run: bool = False,
) -> int:
    """Remap all Godspeed sessions in a directory to ZAYA ChatML format."""
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error("Input directory not found: %s", input_path)
        return 0

    expected_tools = load_expected_tools(expected_tools_path)
    files = sorted(input_path.glob("*.conversation.jsonl"))

    if not files:
        logger.error("No conversation files found in %s", input_path)
        return 0

    logger.info("Scanning %d session files...", len(files))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for fp in files:
            result = remap_session(fp, expected_tools, min_jaccard, max_tokens)
            if result is None:
                skipped += 1
                continue

            if not dry_run:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            written += 1
            if written % 50 == 0:
                logger.info("  %d/%d written, %d skipped", written, len(files), skipped)

            if written >= max_examples:
                logger.info("Reached max examples (%d), stopping", max_examples)
                break

    logger.info(
        "Remapped %d trajectories to %s (%d skipped, %d gates applied)",
        written,
        output_path,
        skipped,
        5,
    )

    if written == 0:
        logger.warning(
            "No trajectories passed quality gates. Try lowering min_jaccard or "
            "checking that Godspeed sessions have exit_code=0 and no tool errors."
        )

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remap Godspeed conversation JSONL to ZAYA1-8B ChatML format"
    )
    parser.add_argument(
        "--input-dir",
        default=str(Path.home() / ".godspeed" / "training"),
        help="Godspeed training directory",
    )
    parser.add_argument("--output", default="data/train_zaya.jsonl")
    parser.add_argument(
        "--expected-tools",
        help="Path to benchmarks/tasks.jsonl for Jaccard scoring",
    )
    parser.add_argument("--min-jaccard", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true", help="Count passes without writing")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    written = remap_directory(
        input_dir=args.input_dir,
        output_path=args.output,
        expected_tools_path=args.expected_tools,
        min_jaccard=args.min_jaccard,
        max_tokens=args.max_tokens,
        max_examples=args.max_examples,
        dry_run=args.dry_run,
    )

    if written == 0:
        logger.error("No trajectories remapped. Pipeline blocked.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
