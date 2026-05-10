"""vLLM server launcher for ZAYA1-8B.

Starts an OpenAI-compatible API server using the Zyphra vLLM fork.
Requires vLLM built from https://github.com/Zyphra/vllm/tree/zaya1-pr

Usage:
    python scripts/serve.py [--port 8010] [--max-model-len 32000]

The server exposes /v1/chat/completions compatible with LiteLLM's
openai/ provider. Point Godspeed at http://localhost:8010/v1.

For 16 GB GPUs, use --max-model-len 24000–32000.
For 24 GB GPUs, use --max-model-len 48000–65536.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="ZAYA1-8B vLLM server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("ZAYA_PORT", "8010")))
    parser.add_argument(
        "--model-id",
        default=os.environ.get("ZAYA_MODEL_ID", "Zyphra/ZAYA1-8B"),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("ZAYA_MAX_MODEL_LEN", "32000")),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=int(os.environ.get("ZAYA_MAX_NUM_SEQS", "2")),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("ZAYA_GPU_MEMORY", "0.95")),
    )
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "vllm", "serve", args.model_id,
        "--port", str(args.port),
        "--mamba-cache-dtype", "float32",
        "--dtype", "bfloat16",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "zaya_xml",
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]

    print(f"Starting vLLM server: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
