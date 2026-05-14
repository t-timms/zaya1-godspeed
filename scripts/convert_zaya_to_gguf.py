#!/usr/bin/env python3
"""Convert ZAYA1-8B HF safetensors to GGUF FP16 format.

Output GGUF is ready for NVFP4 quantization via llama.cpp's llama-quantize,
and for loading via vLLM's GGUF loader.

Usage:
    python scripts/convert_zaya_to_gguf.py [--output /path/to/output.gguf]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def find_snapshot(model_id: str = "Zyphra/ZAYA1-8B") -> Path:
    base = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshots = base / "snapshots"
    if not snapshots.exists():
        print(f"Model not cached at {base}")
        sys.exit(1)
    snaps = sorted(snapshots.iterdir())
    return snaps[-1]


def load_config(snapshot: Path) -> dict:
    with open(snapshot / "config.json") as f:
        return json.load(f)


def load_index(snapshot: Path) -> dict:
    with open(snapshot / "model.safetensors.index.json") as f:
        return json.load(f)


def shorten_name(name: str) -> str:
    """Shorten Zaya HF tensor names to fit GGUF 64-char limit.

    Mappings:
        zaya_block -> zblk
        local_experts -> lexp
        linear_fc -> fc
        self_attn -> attn
        input_norm -> inp_n
        res_scale -> rs
        hidden_states -> hs
        residual -> res
        router_mlp -> rmlp
        balancing_biases -> bal
        rmsnorm_eda -> rms_e
        router_states_scale -> rss
    """
    replacements = [
        (".zaya_block.", ".zblk."),
        (".local_experts.", ".lexp."),
        (".linear_fc", ".fc"),
        (".self_attn.", ".attn."),
        (".input_norm.", ".inp_n."),
        (".res_scale.", ".rs."),
        (".hidden_states", ".hs"),
        (".residual", ".res"),
        (".router_mlp.", ".rmlp."),
        (".balancing_biases", ".bal"),
        (".rmsnorm_eda.", ".rms_e."),
        (".router_states_scale", ".rss"),
    ]
    result = name
    for old, new in replacements:
        result = result.replace(old, new)
    if len(result) >= 64:
        raise ValueError(f"Name still too long after shortening: {len(result)} chars: {result}")
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert ZAYA1-8B to GGUF")
    parser.add_argument("--output", default="/tmp/zaya1-8b-f16.gguf")
    parser.add_argument("--model-id", default="Zyphra/ZAYA1-8B")
    parser.add_argument(
        "--arch",
        default="zaya",
        help="GGUF architecture (use llama for llama.cpp quantize compat)",
    )
    args = parser.parse_args()

    try:
        from gguf import GGUFWriter
    except ImportError:
        print("Error: gguf library not found. Install: pip install gguf")
        sys.exit(1)

    snapshot = find_snapshot(args.model_id)
    print(f"Snapshot: {snapshot}")

    config = load_config(snapshot)
    index = load_index(snapshot)
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    total_tensors = len(weight_map)
    print(f"Tensors: {total_tensors} | Shards: {len(shard_files)}")

    hidden_size = config.get("hidden_size", 2048)
    num_layers = config.get("num_hidden_layers", 80)
    num_heads = config.get("num_attention_heads", 16)
    num_kv_heads = config.get("num_key_value_heads", 2)
    intermediate_size = config.get("intermediate_size", 5632)
    max_pos = config.get("max_position_embeddings", 131072)
    rms_eps = config.get("rms_norm_eps", 1e-6)
    rope_theta = config.get("rope_theta", 5000000.0)
    num_experts = config.get("num_local_experts", 16)
    num_experts_per_tok = config.get("num_experts_per_tok", 1)

    output_path = Path(args.output)
    writer = GGUFWriter(str(output_path), args.arch)

    ap = args.arch
    writer.add_uint32(f"{ap}.block_count", num_layers)
    writer.add_uint32(f"{ap}.context_length", max_pos)
    writer.add_uint32(f"{ap}.embedding_length", hidden_size)
    writer.add_uint32(f"{ap}.feed_forward_length", intermediate_size)
    writer.add_uint32(f"{ap}.attention.head_count", num_heads)
    writer.add_uint32(f"{ap}.attention.head_count_kv", num_kv_heads)
    writer.add_float32(f"{ap}.attention.layer_norm_rms_epsilon", rms_eps)
    writer.add_float32(f"{ap}.rope.freq_base", rope_theta)
    writer.add_uint32(f"{ap}.expert_count", num_experts)
    writer.add_uint32(f"{ap}.expert_used_count", num_experts_per_tok)

    tensor_count = 0
    name_map: dict[str, str] = {}  # shortened -> original

    for shard_file in shard_files:
        shard_path = snapshot / shard_file
        tensors_in_shard = [n for n, s in weight_map.items() if s == shard_file]
        print(f"  {shard_file}: {len(tensors_in_shard)} tensors")

        with safe_open(str(shard_path), framework="pt") as sf:
            for orig_name in tensors_in_shard:
                tensor = sf.get_tensor(orig_name)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)
                elif tensor.dtype not in (torch.float16, torch.float32):
                    tensor = tensor.to(torch.float16)

                short_name = shorten_name(orig_name)
                name_map[short_name] = orig_name
                writer.add_tensor(short_name, tensor.numpy())
                tensor_count += 1

    mapping_path = output_path.with_suffix(".name_map.json")
    with open(mapping_path, "w") as f:
        json.dump(name_map, f, indent=2)
    print(f"Name map saved to {mapping_path}")

    print(f"Writing {tensor_count} tensors to {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_gb = output_path.stat().st_size / (1024**3)
    print(f"Done: {size_gb:.2f} GiB")


if __name__ == "__main__":
    main()
