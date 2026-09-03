#!/usr/bin/env python3
"""Fix: check output_size instead of prefix for router skip."""

path = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
content = open(path).read()

old = """            # RouterSkip: Zaya router layers have bfloat16 weights
            # (float-quantized) incompatible with WNA16 packed format.
            if "router" in (prefix or ""):
                return UnquantizedLinearMethod()"""

new = """            # SmallDimSkip: layers with output_size < 64 can't use
            # any WNA16 kernel (e.g., Zaya router output_size=17).
            # Their weights are stored as bfloat16 (float-quantized),
            # incompatible with the packed format expected by WNA16.
            if hasattr(layer, "output_size") and layer.output_size < 64:
                return UnquantizedLinearMethod()"""

content = content.replace(old, new)
open(path, "w").write(content)
print("UPDATED to output_size check")
