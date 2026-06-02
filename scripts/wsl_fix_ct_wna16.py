"""Handle None kernel (unsupported small layers) in CT WNA16 create_weights"""

p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py"
with open(p) as f:
    c = f.read()

old = """    kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)"""

new = """    kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)
    if kernel_type is None:
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        layer.quant_method = UnquantizedLinearMethod()
        return"""

c = c.replace(old, new)
with open(p, "w") as f:
    f.write(c)
print("Patched: unquantized fallback for empty kernel")
