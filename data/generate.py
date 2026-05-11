"""Generate training data from Godspeed conversation logs.

Converts Godspeed's JSONL audit trail into instruction-tuning format
for fine-tuning ZAYA1-8B on structured tool calling.

Input: Godspeed session files from ~/.godspeed/
Output: ChatML-formatted JSONL for TRL SFTTrainer

A valid trajectory has:
    1. A system prompt with tool definitions
    2. A user task/request
    3. At least one successful tool call → tool result cycle
    4. A final response
    (5.) All tool calls accepted (not rejected by permission engine)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_sessions(audit_dir: str | None = None) -> list[Path]:
    if audit_dir:
        base = Path(audit_dir)
    else:
        base = Path.home() / ".godspeed"
    training = base / "training"
    if not training.exists():
        logger.error("No Godspeed training dir at %s", training)
        return []
    files = sorted(training.glob("*.conversation.jsonl"))
    logger.info("Found %d Godspeed session files", len(files))
    return files


def parse_conversation(filepath: Path) -> list[dict[str, str]] | None:
    messages: list[dict[str, str]] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = entry.get("role", "")
            content = entry.get("content", "")

            if role == "system":
                messages.append({"role": "system", "content": content})
            elif role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                messages.append({"role": "tool", "content": content})

    if len(messages) < 3:
        return None

    has_tool_call = _has_tool_use(messages)
    if not has_tool_call:
        return None

    has_multiple_turns = sum(1 for m in messages if m["role"] == "assistant") >= 2
    if not has_multiple_turns:
        return None

    return messages


def _has_tool_use(messages: list[dict[str, str]]) -> bool:
    for m in messages:
        if m["role"] == "assistant" and ("<zyphra_tool_call>" in m["content"] or '"name":' in m["content"]):
            return True
    return False


def filter_quality(messages: list[dict[str, str]]) -> bool:
    """Quick filters to keep only high-quality trajectories."""
    for m in messages:
        if m["role"] == "assistant":
            c = m["content"]
            if len(c) < 20:
                return False
            if "I cannot" in c and "tool" in c.lower():
                return False
            if "I'm unable" in c and "tool" in c.lower():
                return False
    return True


def truncate_long_content(messages: list[dict[str, str]], max_chars: int = 8000) -> list[dict[str, str]]:
    """Truncate very long tool outputs to keep training examples manageable."""
    result: list[dict[str, str]] = []
    for m in messages:
        content = m["content"]
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n... [truncated {len(m['content']) - max_chars} chars]"
        result.append({"role": m["role"], "content": content})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training data from Godspeed sessions")
    parser.add_argument("--audit-dir", help="Godspeed data directory (default: ~/.godspeed)")
    parser.add_argument("--output", default="data/train.jsonl")
    parser.add_argument("--max-examples", type=int, default=2000)
    parser.add_argument("--min-examples", type=int, default=100, help="Warn if fewer than this many found")
    parser.add_argument("--no-truncate", action="store_true")
    args = parser.parse_args()

    files = find_sessions(args.audit_dir)
    if not files:
        logger.error("No session files found. Run Godspeed with a strong model first.")
        return

    trajectories: list[dict] = []
    for fp in files:
        msgs = parse_conversation(fp)
        if msgs and filter_quality(msgs):
            if not args.no_truncate:
                msgs = truncate_long_content(msgs)
            trajectories.append({"messages": msgs})
        if len(trajectories) >= args.max_examples:
            break

    logger.info("Extracted %d tool-call trajectories from %d sessions", len(trajectories), len(files))

    if len(trajectories) < args.min_examples:
        logger.warning(
            "Only %d trajectories found (minimum %d). Run more Godspeed sessions.",
            len(trajectories),
            args.min_examples,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in trajectories:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info("Wrote %d examples to %s", len(trajectories), output_path)


if __name__ == "__main__":
    main()
