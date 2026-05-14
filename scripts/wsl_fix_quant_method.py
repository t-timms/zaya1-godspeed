p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py"
with open(p) as f:
    c = f.read()

old = """def _skip_marlin_if_unaligned(layer, size_k, size_n):
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

new = """def _skip_marlin_if_unaligned(layer, size_k, size_n):
    if size_n % 64 != 0 or size_k % 256 != 0:
        import logging
        lg = logging.getLogger(__name__)
        lg.warning("Skipping Marlin repack: size_n=%d size_k=%d (not tile-aligned)", size_n, size_k)
        from vllm.model_executor.layers.quantization.unquantized import UnquantizedLinearMethod
        layer.quant_method = UnquantizedLinearMethod()
        return True
    return False"""

if old in c:
    c = c.replace(old, new)
    with open(p, "w") as f:
        f.write(c)
    print("Fixed: unaligned modules use UnquantizedLinearMethod")
else:
    print("Pattern mismatch - searching...")
    if "weight_packed" in c and "_marlin_repack_skipped" in c:
        print("Found _marlin_repack_skipped references")
    else:
        print("Refs not found")
