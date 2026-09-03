#!/usr/bin/env python3
"""Fix NVFP4 scheme to use configurable group_size."""

# 1. Fix the scheme class to accept group_size
nvfp4_path = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py"
nvfp4 = open(nvfp4_path).read()

old_init = "    def __init__(self):\n        self.group_size = 16"
new_init = "    def __init__(self, group_size: int = 16):\n        self.group_size = group_size"
nvfp4 = nvfp4.replace(old_init, new_init)
open(nvfp4_path, "w").write(nvfp4)
print("1. NVFP4 scheme now accepts group_size parameter")

# 2. Fix _get_scheme_from_parts to pass group_size
ct_path = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
ct = open(ct_path).read()

old_scheme = "            return CompressedTensorsW4A16Fp4()"
new_scheme = "            return CompressedTensorsW4A16Fp4(group_size=weight_quant.group_size)"
ct = ct.replace(old_scheme, new_scheme)

# Also fix the second occurrence (MoE)
old_scheme2 = "                    return CompressedTensorsW4A16Fp4()"
new_scheme2 = "                    return CompressedTensorsW4A16Fp4(group_size=weight_quant.group_size)"
ct = ct.replace(old_scheme2, new_scheme2)

open(ct_path, "w").write(ct)
print("2. _get_scheme_from_parts now passes group_size")
