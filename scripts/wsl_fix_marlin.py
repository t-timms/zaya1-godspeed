p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py"
with open(p) as f:
    c = f.read()

# Add skip function before prepare_fp4_layer_for_marlin
old = "def prepare_fp4_layer_for_marlin("
new = 'def _skip_marlin_if_unaligned(layer, size_k, size_n):\n    if size_n % 64 != 0 or size_k % 256 != 0:\n        import logging\n        lg = logging.getLogger(__name__)\n        lg.warning("Skipping Marlin repack: size_n=%d size_k=%d (not tile-aligned)", size_n, size_k)\n        return True\n    return False\n\ndef prepare_fp4_layer_for_marlin('
c = c.replace(old, new)

# Add skip check before gptq_marlin_repack call
old2 = "marlin_qweight = ops.gptq_marlin_repack("
new2 = """if _skip_marlin_if_unaligned(layer, part_size_k, part_size_n):
        return
    marlin_qweight = ops.gptq_marlin_repack("""
c = c.replace(old2, new2)

with open(p, "w") as f:
    f.write(c)
print("Patched: Marlin repack skips unaligned layers")
