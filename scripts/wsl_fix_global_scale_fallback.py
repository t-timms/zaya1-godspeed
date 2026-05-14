#!/usr/bin/env python3
"""Fix: Don't pass garbage global_scale to dequantize in Python fallback.

Bug: process_weights_after_loading clones _weight_global_scale_data from
weight_global_scale (initialized as torch.empty() in create_weights).
The checkpoint has no weight_global_scale keys, so the clone is
uninitialized GPU memory (often zeros). The apply_weights fallback
passes this to dequantize(global_scale=zeros), which zeroes out all
attention weights → hidden states collapse → pad tokens.

Fix: In apply_weights, check if _weight_global_scale_data is all zeros
or NaN. Skip global_scale if invalid.
"""

from pathlib import Path

SCHEME_PATH = Path(
    "/home/ttimm/vllm-env/lib/python3.12/site-packages/"
    "vllm/model_executor/layers/quantization/compressed_tensors/schemes/"
    "compressed_tensors_w4a16_nvfp4.py"
)


def fix() -> bool:
    content = SCHEME_PATH.read_text()
    modified = False

    # Fix the apply_weights fallback to validate global_scale
    old = (
        "            wgs = getattr(layer, \"_weight_global_scale_data\", None)\n"
        "            m, nh = wq.shape\n"
        "            w = unpack_fp4_from_uint8(wq, m, nh * 2)\n"
        "            w = dequantize(x_q=w, scale=ws.float(), global_scale=wgs, dtype=ws.float().dtype)"
    )
    new = (
        "            wgs = getattr(layer, \"_weight_global_scale_data\", None)\n"
        "            # Skip global_scale if uninitialized (all zeros/NaN from torch.empty).\n"
        "            # Missing from checkpoint (symmetric per-group quant, no per-channel rescale).\n"
        "            if wgs is not None and (wgs.abs().max() < 1e-10 or torch.isnan(wgs).any()):\n"
        "                wgs = None\n"
        "            m, nh = wq.shape\n"
        "            w = unpack_fp4_from_uint8(wq, m, nh * 2)\n"
        "            w = dequantize(x_q=w, scale=ws.float(), global_scale=wgs, dtype=ws.float().dtype)"
    )

    if old in content and "Skip global_scale if uninitialized" not in content:
        content = content.replace(old, new)
        print("  [FIX] Skip garbage global_scale in Python dequant fallback")
        modified = True
    elif "Skip global_scale if uninitialized" in content:
        print("  [OK] Global scale validation already present")
    else:
        print("  [WARN] Pattern not found — may need manual review")
        # Show what's there
        for i, line in enumerate(content.split("\n")):
            if "_weight_global_scale_data" in line:
                print(f"    Line {i}: {line}")

    if modified:
        SCHEME_PATH.write_text(content)
        print("  [SAVED] compressed_tensors_w4a16_nvfp4.py updated")

    return modified


if __name__ == "__main__":
    if not SCHEME_PATH.exists():
        print(f"ERROR: NVFP4 scheme not found at {SCHEME_PATH}")
        exit(1)
    ok = fix()
    if ok:
        print("\nFix applied. Re-run quick check to verify generation quality.")
    else:
        print("\nNo changes needed.")
