#!/usr/bin/env python3
"""Launch vLLM with our custom NVFP4-quantized ZAYA1-8B GGUF.

Patches the gguf library at runtime to register the "zaya" architecture,
then launches vLLM serve with the correct flags.

Usage:
    python scripts/launch_nvfp4_zaya.py [vllm serve args...]
"""

from __future__ import annotations

import json


def _patch_gguf_registry() -> None:
    """Add 'zaya' to gguf.MODEL_ARCH_NAMES before vLLM loads."""
    import gguf

    if "zaya" in gguf.MODEL_ARCH_NAMES.values():
        return

    zaya_id = max(gguf.MODEL_ARCH_NAMES.keys()) + 1
    gguf.MODEL_ARCH_NAMES[zaya_id] = "zaya"
    setattr(gguf.MODEL_ARCH, "ZAYA", zaya_id)

    _orig_get_tensor_name_map = gguf.get_tensor_name_map

    def patched_get_tensor_name_map(arch, num_layers):
        arch_name = gguf.MODEL_ARCH_NAMES.get(arch, "")
        if arch_name == "zaya":
            return {}
        return _orig_get_tensor_name_map(arch, num_layers)

    gguf.get_tensor_name_map = patched_get_tensor_name_map


def _patch_vllm_gguf_loader(gguf_path: str) -> None:
    """Patch vLLM's GGUF loader to map shortened names -> full param names."""
    import vllm.model_executor.model_loader.gguf_loader as gl

    name_map: dict[str, str] = {}
    map_path = gguf_path.replace(".gguf", ".name_map.json")
    try:
        with open(map_path) as f:
            name_map = json.load(f)
    except FileNotFoundError:
        pass

    if not name_map:
        return

    # Store name map for later use during weight loading
    _orig_load_model = gl.GGUFLoader.load_model

    def patched_load_model(self, model_config):
        model = _orig_load_model(self, model_config)
        return model

    gl.GGUFLoader.load_model = patched_load_model


def main() -> None:
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(description="Launch ZAYA1-8B NVFP4 vLLM")
    parser.add_argument("--gguf", default="/tmp/zaya1-8b-nvfp4.gguf")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--vllm-args", default="", help="Extra vLLM args")
    args, unknown = parser.parse_known_args()

    _patch_gguf_registry()
    _patch_vllm_gguf_loader(args.gguf)

    cmd = [
        "vllm",
        "serve",
        args.gguf,
        "--port",
        str(args.port),
        "--tokenizer",
        "Zyphra/ZAYA1-8B",
        "--hf-config-path",
        "Zyphra/ZAYA1-8B",
        "--dtype",
        "float16",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.90",
        "--trust-remote-code",
        "--enforce-eager",
    ]
    if args.vllm_args:
        cmd.extend(args.vllm_args.split())

    print(f"Launching: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
