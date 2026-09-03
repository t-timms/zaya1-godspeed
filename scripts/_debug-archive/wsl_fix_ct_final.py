p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
with open(p) as f:
    c = f.read()

# Fix 1: _is_static_tensor_w8a8
old = "        is_8_bits = weight_quant.num_bits == 8 and input_quant is not None and input_quant.num_bits == 8\n        weight_strategy = ("
new = "        if input_quant is None:\n            return False\n        is_8_bits = weight_quant.num_bits == 8 and input_quant.num_bits == 8\n        weight_strategy = ("
c = c.replace(old, new)

# Fix 2: _is_dynamic_token_w8a8
old = "        is_8_bits = weight_quant.num_bits == 8 and input_quant is not None and input_quant.num_bits == 8\n        weight_strategy = (\n            weight_quant.strategy == QuantizationStrategy.TENSOR.value\n            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value\n        )\n        is_token = ("
new = "        if input_quant is None:\n            return False\n        is_8_bits = weight_quant.num_bits == 8 and input_quant.num_bits == 8\n        weight_strategy = (\n            weight_quant.strategy == QuantizationStrategy.TENSOR.value\n            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value\n        )\n        is_token = ("
c = c.replace(old, new)

# Fix 3: _is_dynamic_token_w4a8_int
old = "        is_weight_4_bits = weight_quant.num_bits == 4\n        is_activation_8_bits = input_quant.num_bits == 8"
new = "        if input_quant is None:\n            return False\n        is_weight_4_bits = weight_quant.num_bits == 4\n        is_activation_8_bits = input_quant.num_bits == 8"
c = c.replace(old, new)

with open(p, "w") as f:
    f.write(c)
print("All 3 methods patched with input_quant is None guards")
