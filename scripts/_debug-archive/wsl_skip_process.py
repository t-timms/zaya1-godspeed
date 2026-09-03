p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
with open(p) as f:
    c = f.read()

old = """        if hasattr(layer, "weight_packed"):
            layer.scheme.process_weights_after_loading(layer)"""
new = """        pass  # weights in correct int32 format for WNA16 Marlin"""

c = c.replace(old, new)
with open(p, "w") as f:
    f.write(c)
print("Done - process_weights_after_loading skipped")
