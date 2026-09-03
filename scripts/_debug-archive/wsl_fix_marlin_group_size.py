#!/usr/bin/env python3
"""Fix: Skip Marlin repack for unsupported group_size (64) and fall back to Python dequant.

Bug: CompressedTensorsW4A16Fp4.process_weights_after_loading calls
prepare_fp4_layer_for_marlin() which hardcodes group_size=16. When the
model uses group_size=64, the Marlin kernel fails at inference with
"Invalid thread config" because the fp4 kernel only supports group_size=16.

Fix: Check self.group_size against FP4_MARLIN_SUPPORTED_GROUP_SIZES.
If unsupported, set _marlin_repack_skipped=True to trigger the
Python dequant fallback in apply_weights.
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

    # ── Add group_size check before calling prepare_fp4_layer_for_marlin ──
    # The existing process_weights_after_loading calls prepare_fp4_layer_for_marlin(layer)
    # unconditionally. We need to add a group_size check.
    old_call = "        prepare_fp4_layer_for_marlin(layer)"
    new_call = (
        "        # Check if group_size is supported by the Marlin fp4 kernel.\n"
        "        # Only group_size=16 is supported natively; fall back to Python\n"
        "        # dequant for other sizes (e.g., group_size=64 from CT config).\n"
        "        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (  # noqa: E501\n"
        "            FP4_MARLIN_SUPPORTED_GROUP_SIZES,\n"
        "        )\n"
        "        if self.group_size not in FP4_MARLIN_SUPPORTED_GROUP_SIZES:\n"
        "            import logging\n"
        "            _lg = logging.getLogger(__name__)\n"
        "            _lg.warning(\n"
        '                "Skipping Marlin repack: group_size=%d not in "\n'
        '                "FP4_MARLIN_SUPPORTED_GROUP_SIZES=%s. "\n'
        '                "Falling back to Python dequant.",\n'
        "                self.group_size, FP4_MARLIN_SUPPORTED_GROUP_SIZES,\n"
        "            )\n"
        "            layer._marlin_repack_skipped = True\n"
        "            return\n"
        "\n"
        "        prepare_fp4_layer_for_marlin(layer)"
    )

    if old_call in content and "FP4_MARLIN_SUPPORTED_GROUP_SIZES" not in content:
        content = content.replace(old_call, new_call)
        print("  [FIX] Added group_size check before prepare_fp4_layer_for_marlin")
        modified = True
    elif "FP4_MARLIN_SUPPORTED_GROUP_SIZES" in content:
        print("  [OK] Group size check already present")
    else:
        print("  [WARN] Pattern not found — may need manual review")

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
        print("\nFix applied. Re-run smoke test to verify.")
    else:
        print("\nNo changes needed or fix was not applied.")
