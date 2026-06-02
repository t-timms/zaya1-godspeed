p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py"
with open(p) as f:
    c = f.read()

old = """def _skip_marlin_if_unaligned(layer, size_k, size_n):
    if size_n % 64 != 0 or size_k % 256 != 0:
        import logging
        lg = logging.getLogger(__name__)
        lg.warning("Skipping Marlin repack: size_n=%d size_k=%d (not tile-aligned)", size_n, size_k)
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
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

new = """def _skip_marlin_if_unaligned(layer, size_k, size_n):
    if not hasattr(layer, "weight_packed"):
        return True  # Not a CT-quantized layer, skip Marlin
    if size_n % 64 != 0 or size_k % 256 != 0:
        import logging
        lg = logging.getLogger(__name__)
        lg.warning("Decompressing unaligned module: size_n=%d size_k=%d", size_n, size_k)
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
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
    print("Fixed: check weight_packed exists first")
else:
    print("Pattern not found - re-reading...")
    # Check what's actually there
    lines = c.split("\n")
    for i, line in enumerate(lines):
        if "def _skip_marlin_if_unaligned" in line:
            print("Found at line", i)
            for j in range(i, min(i + 15, len(lines))):
                print(f"  {j}: {lines[j]}")
            break
