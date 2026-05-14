import re

# 1. Remove debug prints from zaya.py
p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
with open(p) as f:
    c = f.read()
c = re.sub(r'\s*logger\.warning\("DEBUG.*?".*?\n', '', c)
with open(p, "w") as f:
    f.write(c)

# 2. Skip process_weights_after_loading for unquantized layers
p2 = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
with open(p2) as f:
    c2 = f.read()

old = "layer.scheme.process_weights_after_loading(layer)"
new = """if getattr(layer, "quant_method", None) is None or not isinstance(layer.quant_method, UnquantizedLinearMethod):
                layer.scheme.process_weights_after_loading(layer)"""

if old in c2:
    c2 = c2.replace(old, new)
    # Add import if missing
    if "from vllm.model_executor.layers.linear import UnquantizedLinearMethod" not in c2:
        c2 = c2.replace(
            "from vllm.model_executor.layers.quantization.compressed_tensors.schemes",
            "from vllm.model_executor.layers.linear import UnquantizedLinearMethod\nfrom vllm.model_executor.layers.quantization.compressed_tensors.schemes"
        )
    with open(p2, "w") as f:
        f.write(c2)
    print("Patched: skip process_weights_after_loading for unquantized layers")
else:
    print("Pattern not found")
