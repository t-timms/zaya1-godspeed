#!/usr/bin/env python3
"""Remove input_quant=None requirement for NVFP4 scheme selection."""

path = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
content = open(path).read()

old = "        if self._is_nvfp4_format(weight_quant) and input_quant is None:"
new = "        if self._is_nvfp4_format(weight_quant):"
content = content.replace(old, new)

# Also fix the moe method check (line ~663)
old2 = "            if self._is_nvfp4_format(weight_quant) and self._is_nvfp4_format("
new2 = "            if self._is_nvfp4_format(weight_quant) and self._is_nvfp4_format("
# That one already doesn't have input_quant check

open(path, "w").write(content)
print("REMOVED input_quant=None requirement")
