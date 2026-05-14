#!/usr/bin/env python3
"""Launch ZAYA1-8B NVFP4 via vLLM Python API.

Bypasses CLI architecture validation by registering ZayaForCausalLM
before creating the LLM engine.
"""

from __future__ import annotations

import json
import os
import sys


def main():
    gguf_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/zaya1-8b-nvfp4.gguf"

    # Register ZayaForCausalLM in vLLM
    from vllm.model_executor.models.registry import ModelRegistry

    ModelRegistry.register_model(
        "ZayaForCausalLM",
        "vllm.model_executor.models.zaya:ZayaForCausalLM",
    )
    print("[ZAYA] Registered ZayaForCausalLM")

    # Patch gguf for zaya arch
    import gguf

    if "zaya" not in gguf.MODEL_ARCH_NAMES.values():
        zid = max(gguf.MODEL_ARCH_NAMES.keys()) + 1
        gguf.MODEL_ARCH_NAMES[zid] = "zaya"
        print("[ZAYA] Registered zaya in gguf")

    # Load name map
    name_map = {}
    map_path = gguf_path.replace(".gguf", ".name_map.json")
    if os.path.exists(map_path):
        with open(map_path) as f:
            name_map = json.load(f)
        print(f"[ZAYA] Loaded name map: {len(name_map)} entries")

    # Launch vLLM server using subprocess (same as CLI but with PYTHONSTARTUP)
    import subprocess

    env = os.environ.copy()
    env["PYTHONSTARTUP"] = "/root/vllm-env/lib/python3.12/site-packages/zaya_startup.py"

    cmd = [
        "vllm",
        "serve",
        gguf_path,
        "--port",
        "8010",
        "--tokenizer",
        "Zyphra/ZAYA1-8B",
        "--hf-config-path",
        "Zyphra/ZAYA1-8B",
        "--dtype",
        "float16",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.90",
        "--trust-remote-code",
        "--enforce-eager",
    ]
    print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    main()
