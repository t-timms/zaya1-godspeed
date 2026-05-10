"""ZAYA1-8B vLLM server for Godspeed integration.

Starts the Zyphra vLLM server with tool-calling support
and Godspeed-compatible settings.

Usage:
    # Start server (Windows via WSL)
    python scripts/serve_zaya1.py

    # Point Godspeed at it:
    export OPENAI_BASE_URL=http://localhost:8010/v1
    export OPENAI_API_KEY=not-needed
    godspeed --model openai/zaya1-8b
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="ZAYA1-8B vLLM server for Godspeed")
    parser.add_argument("--port", type=int, default=int(os.environ.get("ZAYA_PORT", "8010")))
    parser.add_argument(
        "--model-id",
        default=os.environ.get("ZAYA_MODEL_ID", "Zyphra/ZAYA1-8B"),
    )
    parser.add_argument("--max-model-len", type=int, default=24000)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--gpu-memory", type=float, default=0.90)
    parser.add_argument("--speculative-model", default="[ngram]", help="N-gram speculation for 1.2x speedup")
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    args = parser.parse_args()

    cmd = [
        "wsl", "-d", "Ubuntu", "--", "bash", "-c",
        f"source $HOME/vllm-env/bin/activate && "
        f"vllm serve {args.model_id} "
        f"--port {args.port} "
        f"--dtype bfloat16 "
        f"--reasoning-parser qwen3 "
        f"--enable-auto-tool-choice "
        f"--tool-call-parser zaya_xml "
        f"--max-model-len {args.max_model_len} "
        f"--max-num-seqs {args.max_num_seqs} "
        f"--gpu-memory-utilization {args.gpu_memory} "
        f"--speculative-model '{args.speculative_model}' "
        f"--num-speculative-tokens {args.num_speculative_tokens} "
        f"--trust-remote-code",
    ]

    print(f"Starting ZAYA1-8B vLLM server on port {args.port}")
    print(f"Max context: {args.max_model_len} | N-gram speculation: {args.num_speculative_tokens} tokens")
    print(f"Run: {' '.join(cmd)}")
    print()
    print("Godspeed: export OPENAI_BASE_URL=http://localhost:8010/v1")
    print()

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
