p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py"
with open(p) as f:
    c = f.read()

old = """        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        layer.quant_method = UnquantizedLinearMethod()
        return True
    return False"""

new = """        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
        from compressed_tensors.quantization.lifecycle.forward import dequantize
        import torch.nn as _nn
        wq = layer.weight_packed.data
        ws = layer.weight_scale.data
        gs = getattr(layer, "weight_global_scale", None)
        gs_data = gs.data if gs is not None else None
        m, nh = wq.shape
        w = unpack_fp4_from_uint8(wq, m, nh * 2)
        dw = dequantize(x_q=w, scale=ws.float(), global_scale=gs_data, dtype=ws.float().dtype)
        layer.weight = _nn.Parameter(dw, requires_grad=False)
        layer.quant_method = UnquantizedLinearMethod()
        return True
    return False"""

if old in c:
    c = c.replace(old, new)
    with open(p, "w") as f:
        f.write(c)
    print("Patched: decompress + Unquantized")
else:
    print("Pattern not found")
