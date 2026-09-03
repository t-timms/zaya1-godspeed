p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
with open(p) as f:
    c = f.read()

# Fix _is_static_tensor_w8a8 to handle weight-only quantization (input_quant=None)
old = "def _is_static_tensor_w8a8(self, weight_quant, input_quant):"
new = """def _is_static_tensor_w8a8(self, weight_quant, input_quant):
        if input_quant is None:
            return False  # weight-only quantization"""
c = c.replace(old, new)

with open(p, "w") as f:
    f.write(c)
print("Patched: _is_static_tensor_w8a8 handles input_quant=None")
