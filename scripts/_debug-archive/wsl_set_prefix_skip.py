#!/usr/bin/env python3
"""Set CT get_quant_method to use prefix-based router skip."""

path = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
content = open(path).read()

# Change output_size check back to prefix check
old = """            # SmallDimSkip: layers with output_size < 64 can't use
            # any WNA16 kernel (e.g., Zaya router output_size=17).
            # Their weights are stored as bfloat16 (float-quantized),
            # incompatible with the packed format expected by WNA16.
            if hasattr(layer, "output_size") and layer.output_size < 64:
                return UnquantizedLinearMethod()"""

new = """            # RouterSkip: Zaya router layers have bfloat16 weights
            # (float-quantized) incompatible with WNA16 packed format.
            if "router" in (prefix or ""):
                return UnquantizedLinearMethod()"""

if "RouterSkip" not in content:
    content = content.replace(old, new)
    open(path, "w").write(content)
    print("SET prefix-based router skip")
else:
    print("RouterSkip already present")

# Also verify the fix
check = open(path).read()
if '"router" in (prefix' in check:
    print("VERIFIED: prefix check present")
