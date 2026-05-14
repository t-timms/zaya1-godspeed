#!/usr/bin/env python3
"""Add router prefix check to CT get_quant_method."""

path = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
content = open(path).read()

old = "        if isinstance(layer, LinearBase):\n            # collect schemes"
new = """        if isinstance(layer, LinearBase):
            # RouterSkip: Zaya router layers have bfloat16 weights
            # (float-quantized) incompatible with WNA16 packed format.
            if "router" in (prefix or ""):
                return UnquantizedLinearMethod()
            # collect schemes"""

if "RouterSkip" not in content:
    content = content.replace(old, new)
    open(path, "w").write(content)
    print("ADDED RouterSkip check")
else:
    print("ALREADY PRESENT")
