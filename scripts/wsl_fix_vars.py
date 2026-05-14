p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py"
with open(p) as f:
    c = f.read()
c = c.replace('getattr(layer, "_marlin_repack_skipped", False)', 'getattr(weight, "_marlin_repack_skipped", False)')
c = c.replace('x_q, x_s, x_zp, x_gs = layer.weight_packed, layer.weight_scale, None, getattr(layer, "weight_global_scale", None)', 'x_q, x_s, x_zp, x_gs = weight, weight_scale, None, weight_global_scale')
c = c.replace('output = torch.nn.functional.linear(x, weight.to(x.dtype))', 'output = torch.nn.functional.linear(input, w.to(input.dtype))')
with open(p, "w") as f:
    f.write(c)
print("Fixed")
