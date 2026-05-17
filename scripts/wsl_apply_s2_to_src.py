#!/usr/bin/env python3
"""Run session 2 fix script against vllm-src (editable install)."""
import sys
sys.path.insert(0, "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/scripts")

# Monkey-patch the VLLM path before importing
import wsl_fix_nvfp4_text_gen as fix
import pathlib
fix.VLLM = pathlib.Path("/home/ttimm/vllm-src/vllm")
fix.ZAYA = fix.VLLM / "model_executor/models/zaya.py"
fix.MOE = (
    fix.VLLM
    / "model_executor/layers/quantization/compressed_tensors/"
    / "compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py"
)

print(f"Target ZAYA: {fix.ZAYA}")
print(f"Target MOE: {fix.MOE}")
print()

fix.main()
