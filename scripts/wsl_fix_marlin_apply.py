"""SOTA fix: Marlin fallback for unaligned CCA router MLP layers.
Sets flag on nn.Module during repack-skip, checks it in CT scheme apply_weights.
"""
# Fix 1: Ensure flag is set on layer._marlin_repack_skipped in _skip_marlin_if_unaligned
# (already done in marlin_utils_fp4.py)

# Fix 2: Patch compressed_tensors_w4a16_nvfp4.py apply_weights to check flag
p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py"
with open(p) as f:
    c = f.read()

old = """    @override
    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return apply_fp4_marlin_linear("""
new = """    @override
    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if getattr(layer, "_marlin_repack_skipped", False):
            # Unaligned module (CCA router MLP, size_n=17) - fall back to non-Marlin
            from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
            from compressed_tensors.quantization.lifecycle.forward import dequantize
            wq = layer.weight_packed
            ws = layer.weight_scale
            gs = getattr(layer, "weight_global_scale", None)
            m, nh = wq.shape
            w = unpack_fp4_from_uint8(wq, m, nh * 2)
            w = dequantize(x_q=w, scale=ws.float(), global_scale=gs, dtype=ws.float().dtype)
            output = torch.nn.functional.linear(x, w.to(x.dtype))
            return output + bias if bias is not None else output

        return apply_fp4_marlin_linear("""
c = c.replace(old, new)

# Also clean up the broken marlin_utils_fp4.py fallback (which can't work due to tensor attribute loss)
p2 = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py"
with open(p2) as f:
    c2 = f.read()
# Remove our broken inline fallback from marlin_utils_fp4.py
old_fb = """    if getattr(weight, "_marlin_repack_skipped", False) or getattr(weight_scale, "_marlin_repack_skipped", False):
        # Fall back to non-Marlin dequant for unaligned modules (CCA router)
        from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
        from compressed_tensors.quantization.lifecycle.forward import dequantize
        m, n_half = weight.shape
        w = unpack_fp4_from_uint8(weight, m, n_half * 2)
        w = dequantize(x_q=w, scale=weight_scale.float(), global_scale=weight_global_scale, dtype=weight_scale.float().dtype)
        output = torch.nn.functional.linear(input, w.to(input.dtype))
        return output + bias if bias is not None else output

    output = ops.marlin_gemm("""
new_fb = """    output = ops.marlin_gemm("""
if old_fb in c2:
    c2 = c2.replace(old_fb, new_fb)
    with open(p2, "w") as f:
        f.write(c2)
    print("Cleaned up broken fallback in marlin_utils_fp4.py")

with open(p, "w") as f:
    f.write(c)
print("Patched: CT W4A16Fp4 apply_weights falls back for unaligned modules")
