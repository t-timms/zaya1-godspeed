p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py"
with open(p) as f:
    c = f.read()

old = """    ) -> torch.Tensor:
        return apply_fp4_marlin_linear("""
new = """    ) -> torch.Tensor:
        if getattr(layer, "_marlin_repack_skipped", False):
            from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
            from compressed_tensors.quantization.lifecycle.forward import dequantize
            import torch as _t
            wq, ws, gs = layer.weight, layer.weight_scale, getattr(layer, "weight_global_scale", None)
            m, nh = wq.shape
            w = unpack_fp4_from_uint8(wq, m, nh * 2)
            w = dequantize(x_q=w, scale=ws.float(), global_scale=gs, dtype=ws.float().dtype)
            out = _t.nn.functional.linear(x, w.to(x.dtype))
            return out + bias if bias is not None else out
        return apply_fp4_marlin_linear("""
c = c.replace(old, new)
with open(p, "w") as f:
    f.write(c)
print("Patched")
