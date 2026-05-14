"""Fix Marlin inference for modules whose repack was skipped (size_n not tile-aligned).

Module has size_n=17 (CCA router MLP, in_features=34 packed → 17), which Marlin can't handle.
Set a flag on layer.weight_packed so apply_fp4_marlin_linear can fall back to non-Marlin dequant.
"""
p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py"
with open(p) as f:
    c = f.read()

# Fix 1: Update _skip_marlin_if_unaligned to set flag on weight_packed
old = """def _skip_marlin_if_unaligned(layer, size_k, size_n):
    if size_n % 64 != 0 or size_k % 256 != 0:
        import logging
        lg = logging.getLogger(__name__)
        lg.warning("Skipping Marlin repack: size_n=%d size_k=%d (not tile-aligned)", size_n, size_k)
        layer._marlin_repack_skipped = True
        return True
    layer._marlin_repack_skipped = False
    return False"""
new = """def _skip_marlin_if_unaligned(layer, size_k, size_n):
    if size_n % 64 != 0 or size_k % 256 != 0:
        import logging
        lg = logging.getLogger(__name__)
        lg.warning("Skipping Marlin repack: size_n=%d size_k=%d (not tile-aligned)", size_n, size_k)
        if hasattr(layer, "weight_packed"):
            layer.weight_packed._marlin_repack_skipped = True
        layer._marlin_repack_skipped = True
        return True
    layer._marlin_repack_skipped = False
    return False"""
c = c.replace(old, new)

# Fix 2: Add fallback in apply_fp4_marlin_linear
old2 = "    output = ops.marlin_gemm("
new2 = """    if getattr(weight, "_marlin_repack_skipped", False) or getattr(weight_scale, "_marlin_repack_skipped", False):
        # Fall back to non-Marlin dequant for unaligned modules (CCA router)
        from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
        from compressed_tensors.quantization.lifecycle.forward import dequantize
        m, n_half = weight.shape
        w = unpack_fp4_from_uint8(weight, m, n_half * 2)
        w = dequantize(x_q=w, scale=weight_scale.float(), global_scale=weight_global_scale, dtype=weight_scale.float().dtype)
        output = torch.nn.functional.linear(input, w.to(input.dtype))
        return output + bias if bias is not None else output

    output = ops.marlin_gemm("""
c = c.replace(old2, new2)

# Need torch import in the file
if 'import torch' not in c[:100]:
    # Find the first import and add torch
    c = c.replace("from typing import", "import torch\nfrom typing import")

with open(p, "w") as f:
    f.write(c)
print("Patched: Marlin inference falls back for unaligned modules")
