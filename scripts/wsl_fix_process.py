"""Direct fix: skip process_weights_after_loading for modules without weight_packed"""
p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
with open(p) as f:
    c = f.read()

old = """        if getattr(layer, "quant_method", None) is None or not isinstance(layer.quant_method, UnquantizedLinearMethod):
                layer.scheme.process_weights_after_loading(layer)"""

new = """        if hasattr(layer, "weight_packed"):
            layer.scheme.process_weights_after_loading(layer)"""

if old in c:
    c = c.replace(old, new)
    with open(p, "w") as f:
        f.write(c)
    print("Fixed: only process CT layers with weight_packed")
else:
    print("Pattern not found in file")
    # Try finding the line
    for i, line in enumerate(c.split("\n")):
        if "process_weights_after_loading" in line and "scheme" in line:
            print(f"Found at line {i}: {line}")
