"""Generate training data from Godspeed conversation logs.

Converts Godspeed's JSONL audit trail into instruction-tuning format
for fine-tuning ZAYA1-8B on structured tool calling.

Godspeed logs conversations with:
- Per-step reward annotations (accepted vs rejected tool calls)
- Full tool schemas in system prompts
- Tool call arguments and results

This script extracts trajectories where tool calls succeed cleanly
and converts them to: system_prompt → plan → tool_call → result → ...

Usage:
    python data/generate.py --input ~/.godspeed/sessions/*.jsonl --output data/train.jsonl

Output format (sharegpt / chatml):
    {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...<tool_call>...</tool_call>..."},
        {"role": "tool", "content": "..."},
        ...
    ]}
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_sessions(glob_pattern: str) -> list[dict]:
    """Load Godspeed session files from a glob pattern."""
    from glob import glob

    sessions: list[dict] = []
    for path in sorted(glob(glob_pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sessions.append(json.loads(line))
    return sessions


def extract_tool_trajectories(sessions: list[dict]) -> list[dict]:
    """Extract trajectories with successful tool calls.

    A successful trajectory has:
    - At least one tool call
    - No rejected/retried tool calls (or retries are counted separately)
    - A final answer or task completion
    """
    trajectories: list[dict] = []
    # TODO: implement extraction based on Godspeed's reward annotation format
    # This is a placeholder — the actual format depends on Godspeed's audit schema.
    logger.warning("extract_tool_trajectories is a stub — implement based on Godspeed session format")
    return trajectories


def convert_to_sharegpt(trajectories: list[dict]) -> list[dict]:
    """Convert extracted trajectories to ShareGPT/ChatML format."""
    # TODO: implement conversion
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training data from Godspeed logs")
    parser.add_argument("--input", required=True, help="Glob pattern for Godspeed session files")
    parser.add_argument("--output", default="data/train.jsonl", help="Output path")
    args = parser.parse_args()

    sessions = load_sessions(args.input)
    logger.info("Loaded %d sessions", len(sessions))

    trajectories = extract_tool_trajectories(sessions)
    logger.info("Extracted %d tool-call trajectories", len(trajectories))

    converted = convert_to_sharegpt(trajectories)
    logger.info("Converted %d examples", len(converted))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in converted:
            f.write(json.dumps(example) + "\n")
    logger.info("Wrote training data to %s", output_path)


if __name__ == "__main__":
    main()
