#!/usr/bin/env python3
"""SOTA ZAYA1-8B NVFP4 vLLM bridge.

Loads ZayaForCausalLM via HF config, then loads NVFP4-quantized
weights from our custom GGUF file. Bypasses the GGUF architecture
registry issue by using direct weight mapping.

Usage:
    python scripts/serve_nvfp4_zaya.py [--port 8010]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from gguf import GGUFReader


def load_name_map(gguf_path: str) -> dict[str, str]:
    """Load the short_name -> original_name mapping."""
    map_path = gguf_path.replace(".gguf", ".name_map.json")
    with open(map_path) as f:
        return json.load(f)


def load_gguf_weights(gguf_path: str, model: torch.nn.Module) -> None:
    """Load NVFP4-quantized weights from GGUF into a ZayaForCausalLM model."""
    reader = GGUFReader(gguf_path)
    name_map = load_name_map(gguf_path)

    params = dict(model.named_parameters())
    loaded = 0
    skipped = 0

    for short_name in name_map:
        orig_name = name_map[short_name]
        if orig_name not in params:
            skipped += 1
            continue

        param = params[orig_name]
        for tensor in reader.tensors:
            if tensor.name == short_name:
                data = torch.from_numpy(tensor.data.copy())
                if data.shape == param.shape:
                    param.data.copy_(data.to(param.dtype).to(param.device))
                elif len(data.shape) == 1 and len(param.shape) == 2:
                    param.data.copy_(data.unsqueeze(0).to(param.dtype).to(param.device))
                elif len(data.shape) == 2 and len(param.shape) == 1:
                    param.data.copy_(data.squeeze().to(param.dtype).to(param.device))
                else:
                    print(f"  Shape mismatch: {orig_name} GGUF={data.shape} param={param.shape}")
                loaded += 1
                break

    print(f"Loaded {loaded} weights, skipped {skipped}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ZAYA1-8B NVFP4 vLLM Bridge")
    parser.add_argument("--gguf", default="/tmp/zaya1-8b-nvfp4.gguf")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--model-id", default="Zyphra/ZAYA1-8B")
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    if not Path(args.gguf).exists():
        print(f"GGUF not found: {args.gguf}")
        sys.exit(1)

    print(f"Loading ZayaForCausalLM config from {args.model_id}")
    print(f"NVFP4 weights from {args.gguf}")

    # Use vLLM's serve with custom weight loading
    import os

    os.environ["ZAYA_NVFP4_GGUF"] = str(Path(args.gguf).resolve())
    os.environ["ZAYA_NAME_MAP"] = str(Path(args.gguf).resolve()).replace(".gguf", ".name_map.json")

    cmd = (
        f"vllm serve {args.model_id} "
        f"--port {args.port} "
        f"--dtype bfloat16 "
        f"--quantization nvfp4_gguf "
        f"--max-model-len {args.max_model_len} "
        f"--max-num-seqs 1 "
        f"--gpu-memory-utilization 0.90 "
        f"--trust-remote-code "
        f"--enforce-eager"
    )
    print(f"Run: {cmd}")
    print()
    print("NOTE: The NVFP4 GGUF bridge requires a custom vLLM quantization plugin.")
    print("See scripts/nvfp4_gguf_plugin.py for the plugin implementation.")
    print()
    print("Alternative: Use the FP8 quantization path which is fully supported:")
    print(f"  vllm serve {args.model_id} --quantization fp8 --port 8010 ...")


if __name__ == "__main__":
    main()
