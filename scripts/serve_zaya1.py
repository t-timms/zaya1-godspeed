"""ZAYA1-8B vLLM server for Godspeed integration.

Starts the Zyphra vLLM server with tool-calling support and
Godspeed-compatible settings. Uses WSL2 for CUDA access.

Matches the official Zyphra deployment command from:
https://huggingface.co/Zyphra/ZAYA1-8B

MXFP4 quantized variant (OsaurusAI/ZAYA1-8B-MXFP4) recommended
for RTX 5070 Ti Blackwell (sm_120) — ~4.2 GB VRAM vs 16.5 GB bf16.

Usage:
    python scripts/serve_zaya1.py                               # print command
    python scripts/serve_zaya1.py --start                       # start + poll health
    python scripts/serve_zaya1.py --model-id OsaurusAI/ZAYA1-8B-MXFP4 --quantization mx_fp4 --start
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request

DEFAULT_MODEL = "Zyphra/ZAYA1-8B"
DEFAULT_PORT = 8010
DEFAULT_MAX_LEN = 2048
DEFAULT_MAX_SEQS = 2
DEFAULT_GPU_MEM = 0.90
DEFAULT_WAIT = 600


def build_vllm_command(args: argparse.Namespace) -> str:
    """Build the vLLM serve command matching Zyphra's official deployment."""
    extra = []
    if args.enforce_eager:
        extra.append("--enforce-eager")
    if args.quantization:
        extra.append(f"--quantization {args.quantization}")

    parts = [
        f"vllm serve {args.model_id}",
        f"--port {args.port}",
        "--mamba-cache-dtype float32",
        "--dtype bfloat16",
        "--reasoning-parser qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser zaya_xml",
        f"--max-model-len {args.max_model_len}",
        f"--max-num-seqs {args.max_num_seqs}",
        f"--gpu-memory-utilization {args.gpu_memory}",
        "--trust-remote-code",
        *extra,
    ]
    return " ".join(parts)


def build_wsl_command(vllm_cmd: str) -> str:
    """Wrap vLLM command for WSL2 execution."""
    return f'source "$HOME/vllm-env/bin/activate" && exec {vllm_cmd}'


def check_health(port: int, timeout: int = 5) -> bool:
    """Check if the vLLM server is healthy."""
    try:
        url = f"http://localhost:{port}/health"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_server(args: argparse.Namespace) -> int:
    """Start vLLM in WSL2 and optionally wait for readiness."""
    vllm_cmd = build_vllm_command(args)
    wsl_cmd = build_wsl_command(vllm_cmd)

    print(f"Starting {args.model_id} on port {args.port}")
    print(f"  Context: {args.max_model_len} | Seqs: {args.max_num_seqs}")
    print(f"  GPU mem util: {args.gpu_memory}")
    if args.enforce_eager:
        print("  Mode: eager (faster startup, no CUDA graph warmup)")
    if args.quantization:
        print(f"  Quantization: {args.quantization}")
    if args.quiet:
        print("  Quiet mode: no server output streamed")

    try:
        proc = subprocess.Popen(
            ["wsl", "-d", "Ubuntu", "--", "bash", "-c", wsl_cmd],
            stdout=subprocess.PIPE if args.quiet else None,
            stderr=subprocess.STDOUT if args.quiet else None,
            text=True,
        )
    except FileNotFoundError:
        print("ERROR: WSL not found. Install WSL2: wsl --install -d Ubuntu")
        return 1

    print(f"  PID: {proc.pid} (Windows host process)")

    if args.daemon:
        print("  Server starting in background.")
        print(f"  Check: curl http://localhost:{args.port}/health")
        return 0

    t0 = time.time()
    deadline = t0 + args.wait

    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"ERROR: vLLM exited with code {proc.returncode}")
            return proc.returncode or 1

        if check_health(args.port, timeout=3):
            elapsed = time.time() - t0
            print(f"  Server ready in {elapsed:.0f}s")
            print(f"  OpenAI endpoint: http://localhost:{args.port}/v1")
            print(f"  Health check:    http://localhost:{args.port}/health")
            print()
            print("  Godspeed: export OPENAI_BASE_URL=http://localhost:8010/v1")
            print("  Test: curl http://localhost:8010/v1/models")
            return 0

        elapsed = time.time() - t0
        print(f"  Waiting... ({elapsed:.0f}s)", end="\r")
        time.sleep(10)

    print()
    print(f"WARNING: Not ready after {args.wait}s.")
    print(f"  It may still be starting — curl http://localhost:{args.port}/health")
    return 0


def print_command(args: argparse.Namespace) -> None:
    """Print the vLLM command for manual execution."""
    vllm_cmd = build_vllm_command(args)
    wsl_cmd = build_wsl_command(vllm_cmd)
    print("Run in WSL2 (or any terminal with the vllm-env):")
    print()
    print("  source ~/vllm-env/bin/activate")
    print(f"  {vllm_cmd}")
    print()
    print("Or from Windows:")
    print(f'  wsl -d Ubuntu -- bash -c "{wsl_cmd}"')
    print()
    print("Godspeed: export OPENAI_BASE_URL=http://localhost:8010/v1")


def main() -> None:
    parser = argparse.ArgumentParser(description="ZAYA1-8B vLLM server for Godspeed")
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start the server (default: print command only)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Start and exit immediately (dont wait for ready)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("ZAYA_PORT", str(DEFAULT_PORT))),
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("ZAYA_MODEL_ID", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_LEN,
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=DEFAULT_MAX_SEQS,
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=DEFAULT_GPU_MEM,
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=True,
        help="Skip CUDA graph warmup (default: on)",
    )
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Use torch.compile warmup (adds ~8 min startup)",
    )
    parser.add_argument(
        "--quantization",
        help="Quantization: fp8, mx_fp4, awq, gptq, etc.",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=DEFAULT_WAIT,
        help="Max seconds to wait for ready",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress server output",
    )
    args = parser.parse_args()

    if args.start:
        sys.exit(start_server(args))
    else:
        print_command(args)


if __name__ == "__main__":
    main()
