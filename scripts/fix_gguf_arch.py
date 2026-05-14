#!/usr/bin/env python3
"""Rewrite the architecture field in a GGUF file.

Usage:
    python scripts/fix_gguf_arch.py input.gguf [new_arch]
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def fix_gguf_arch(input_path: str, new_arch: str) -> None:
    from gguf import GGUFReader, GGUFWriter

    reader = GGUFReader(input_path)
    out_path = input_path + ".tmp"

    writer = GGUFWriter(out_path, new_arch)

    # Copy all KV metadata (except architecture, which is set in constructor)
    arch_keys_seen: set[str] = set()
    for key, field in reader.fields.items():
        if key == "general.architecture":
            continue

        # Get the first value from the field
        parts = field.parts
        n_parts = len(parts)
        types = field.types

        # Rewrite arch-specific keys
        rewritten_key = key
        for old_prefix in ("llama.", "zaya."):
            if key.startswith(old_prefix):
                rewritten_key = f"{new_arch}.{key[len(old_prefix) :]}"
                break

        if rewritten_key in arch_keys_seen:
            continue
        arch_keys_seen.add(rewritten_key)

        # Copy each part
        for i in range(n_parts):
            val = parts[i][0]  # value is in position 0 of the part tuple
            t = types[i]

            if t == 8:  # GGUFValueType.STRING
                writer.add_string(rewritten_key, val)
            elif t in (4, 5):  # UINT32, INT32
                writer.add_uint32(rewritten_key, int(val))
            elif t in (10, 11):  # UINT64, INT64
                writer.add_uint64(rewritten_key, int(val))
            elif t in (14, 15):  # FLOAT32, FLOAT64
                writer.add_float32(rewritten_key, float(val))
            elif t == 9:  # BOOL
                writer.add_bool(rewritten_key, bool(val))
            # Skip complex types — keep only the first occurrence of each key
            break

    # Copy all tensors
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, tensor.data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Atomic replace
    shutil.move(out_path, input_path)
    print(f"Updated {input_path}: arch={new_arch}")
    print(f"  Size: {Path(input_path).stat().st_size / (1024**3):.2f} GiB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix GGUF architecture field")
    parser.add_argument("gguf_path", help="Path to GGUF file")
    parser.add_argument("--arch", default="zaya", help="New architecture name")
    args = parser.parse_args()

    if not Path(args.gguf_path).exists():
        print(f"File not found: {args.gguf_path}")
        sys.exit(1)

    fix_gguf_arch(args.gguf_path, args.arch)
