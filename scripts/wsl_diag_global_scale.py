#!/usr/bin/env python3
"""Instrument the NVFP4 dequant fallback to check global_scale values."""
from __future__ import annotations

import multiprocessing
import sys

multiprocessing.set_start_method("spawn", force=True)

import logging

logging.basicConfig(level=logging.WARNING)

def main() -> int:
    import torch
    from vllm import LLM

    MODEL_DIR = (
        "/mnt/c/Users/ttimm/Documents/Project Portfolio/"
        "zaya1-godspeed/zaya1-8b-nvfp4-ct"
    )

    print("Loading model...")
    llm = LLM(
        model=MODEL_DIR, dtype="float16", max_model_len=256,
        trust_remote_code=True, enforce_eager=True, max_num_seqs=1,
        tokenizer="Zyphra/ZAYA1-8B",
    )

    # Inspect model parameters after loading
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    model = model.model  # ZayaModel

    print("\n=== Global Scale Diagnostics ===\n")

    bad_layers = []
    for name, param in model.named_parameters():
        if "global_scale" not in name and "input_scale" not in name:
            continue
        val = param.data.float()
        stats = {
            "min": val.min().item(),
            "max": val.max().item(),
            "mean": val.mean().item(),
            "zeros": (val.abs() < 1e-10).sum().item(),
            "total": val.numel(),
            "nan": torch.isnan(val).any().item(),
            "inf": torch.isinf(val).any().item(),
            "zr": (val.abs().max() < 1e-10).item(),
        }
        status = "OK" if not stats["zr"] and not stats["nan"] else "BAD"
        print(f"[{status}] {name}: shape={list(param.shape)} "
              f"range=[{stats['min']:.6f}, {stats['max']:.6f}] "
              f"mean={stats['mean']:.6e} "
              f"zeros={stats['zeros']}/{stats['total']} "
              f"nan={stats['nan']} allzero={stats['zr']}")
        if stats["zr"] or stats["nan"]:
            bad_layers.append(name)

    # Also check _weight_global_scale_data on Linear layers
    print("\n=== Fallback Data Diagnostics (attention layers) ===\n")
    for name, module in model.named_modules():
        if not hasattr(module, "_marlin_repack_skipped"):
            continue
        if not module._marlin_repack_skipped:
            continue
        wgs = getattr(module, "_weight_global_scale_data", None)
        if wgs is None:
            print(f"[OK] {name}: no _weight_global_scale_data")
            continue
        val = wgs.float()
        zr = (val.abs().max() < 1e-10).item()
        nan = torch.isnan(val).any().item()
        status = "BAD" if zr or nan else "OK"
        print(f"[{status}] {name}: _weight_global_scale_data "
              f"range=[{val.min():.6e}, {val.max():.6e}] "
              f"mean={val.mean():.6e} nan={nan} allzero={zr}")

    print(f"\nTotal BAD params: {len(bad_layers)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
