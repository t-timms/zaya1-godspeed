p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
with open(p) as f:
    c = f.read()

old = "is_8_bits = weight_quant.num_bits == input_quant.num_bits == 8"
new = "is_8_bits = weight_quant.num_bits == 8 and input_quant is not None and input_quant.num_bits == 8"

if old in c:
    c = c.replace(old, new)
    with open(p, "w") as f:
        f.write(c)
    print("Patched OK")
else:
    # File might be corrupted by nano - check for similar pattern
    if "input_quant.num_bits" in c:
        import re
        c = re.sub(r"weight_quant\.num_bits == input_quant\.num_bits == 8", new, c)
        with open(p, "w") as f:
            f.write(c)
        print("Patched OK (regex)")
    else:
        print("Pattern not found - file may be corrupted")
