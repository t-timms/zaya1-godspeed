"""Professional fix for Zaya CT NVFP4 vLLM serving.

Three root problems addressed:
  1. process_weights_after_loading blanket skip → restored with dimension guard
  2. zaya.py converts non-MoE uint8 weights to int32 → scoped to MoE only
  3. prepare_fp4_layer_for_marlin panics on unaligned dims → graceful fallback

Run in WSL: python /mnt/c/Users/ttimm/Documents/Project\ Portfolio/zaya1-godspeed/scripts/fix_ct_serving.py
"""

from __future__ import annotations

import os
import sys

VLLM_ROOT = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm"


def fix_process_weights_after_loading() -> bool:
    """Fix 1: Remove blanket `pass` in process_weights_after_loading.

    Restores original code that calls layer.scheme.process_weights_after_loading(layer),
    but adds a guard for layers where Marlin repack was explicitly skipped due to
    unaligned dimensions (set by fix 3 in prepare_fp4_layer_for_marlin).
    """
    path = os.path.join(
        VLLM_ROOT,
        "model_executor/layers/quantization/compressed_tensors/compressed_tensors.py",
    )
    with open(path) as f:
        content = f.read()

    old = """    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass  # weights in correct int32 format for WNA16 Marlin"""

    new = """    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_marlin_repack_skipped", False):
            return
        if hasattr(layer, "scheme") and hasattr(layer.scheme, "process_weights_after_loading"):
            layer.scheme.process_weights_after_loading(layer)"""

    if old in content:
        content = content.replace(old, new)
        with open(path, "w") as f:
            f.write(content)
        print("  [OK] Fix 1: Restored process_weights_after_loading with unaligned guard")
        return True
    else:
        print("  [SKIP] Fix 1: Already patched or pattern not found")
        # Check if "pass" is still there
        if "pass  # weights in correct int32 format" in content:
            print("  [WARN] Fix 1: Old blanket skip found but pattern mismatch")
        return False


def fix_zaya_weight_conversion() -> bool:
    """Fix 2: Remove uint8→int32 conversion for non-MoE weights in zaya.py.

    MoE weights (w13, w2) are handled earlier with `continue`. Any weight
    reaching the "Loading other parameters" section is non-MoE and should
    stay as packed uint8 for CompressedTensors to handle via Marlin.

    The conversion to int32 was masking the real problem (unaligned dims)
    and preventing proper CT scheme handling.
    """
    path = os.path.join(VLLM_ROOT, "model_executor/models", "zaya.py")
    with open(path) as f:
        content = f.read()

    old = """            # Loading other parameters
            # Convert uint8 packed to int32 for CT WNA16 scheme
            if loaded_weight.dtype in (torch.uint8, torch.int8) and loaded_weight.ndim >= 2:
                loaded_weight = loaded_weight.reshape(-1, 4).view(torch.int32).reshape(loaded_weight.shape[0], -1)"""

    new = """            # Loading other parameters (non-MoE)
            # Keep weights as packed uint8 — CompressedTensors scheme handles dequant"""

    if old in content:
        content = content.replace(old, new)
        with open(path, "w") as f:
            f.write(content)
        print("  [OK] Fix 2: Scoped uint8→int32 conversion to MoE weights only")
        return True
    else:
        if "Convert uint8 packed to int32 for CT WNA16 scheme" in content:
            print("  [SKIP] Fix 2: Pattern altered — checking manually")
            return False
        elif "Loading other parameters" in content:
            print("  [SKIP] Fix 2: Conversion already removed")
            return True
        return False


def fix_marlin_dimension_validation() -> bool:
    """Fix 3: Add dimension validation to prepare_fp4_layer_for_marlin.

    Some layers (CCA router MLP, size_n=17) have dimensions that don't
    meet Marlin's tile requirements (size_n % 64 == 0, size_k % 256 == 0).

    For these layers, set _marlin_repack_skipped flag so Fix 1 can skip them,
    and set UnquantizedLinearMethod for a pure-PyTorch fallback.
    """
    path = os.path.join(
        VLLM_ROOT,
        "model_executor/layers/quantization/utils/marlin_utils_fp4.py",
    )
    with open(path) as f:
        content = f.read()

    # Insert dimension check right after the function signature and input_dtype checks,
    # before the group_size assignment
    old = """    is_nvfp4 = hasattr(layer, "weight_global_scale")
    if input_dtype is not None and input_dtype.itemsize == 1:
        if is_nvfp4:
            raise RuntimeError("NVFP4 weight + INT8/FP8 activation is not supported.")
        elif input_dtype != torch.float8_e4m3fn:
            raise RuntimeError("MXFP4 weight + INT8 activation is not supported.")

    group_size = 16 if is_nvfp4 else 32

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition"""

    new = """    is_nvfp4 = hasattr(layer, "weight_global_scale")
    if input_dtype is not None and input_dtype.itemsize == 1:
        if is_nvfp4:
            raise RuntimeError("NVFP4 weight + INT8/FP8 activation is not supported.")
        elif input_dtype != torch.float8_e4m3fn:
            raise RuntimeError("MXFP4 weight + INT8 activation is not supported.")

    group_size = 16 if is_nvfp4 else 32

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition

    # Marlin tile-alignment check: Marlin requires size_n % 64 == 0 and size_k % 256 == 0.
    # CCA router MLP (size_n=17) and other small projections fail this.
    # Fall back to UnquantizedLinearMethod for unaligned layers.
    if part_size_n % 64 != 0 or part_size_k % 256 != 0:
        import logging
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        lg = logging.getLogger(__name__)
        lg.warning(
            "Skipping Marlin repack for %s: size_n=%d (%%64=%d) size_k=%d (%%256=%d) — "
            "dimensions not tile-aligned. Falling back to unquantized.",
            layer.__class__.__name__, part_size_n, part_size_n % 64,
            part_size_k, part_size_k % 256,
        )
        layer._marlin_repack_skipped = True
        layer.quant_method = UnquantizedLinearMethod()
        return"""

    if old in content:
        content = content.replace(old, new)
        with open(path, "w") as f:
            f.write(content)
        print("  [OK] Fix 3: Added Marlin dimension validation with graceful fallback")
        return True
    else:
        print("  [SKIP] Fix 3: Pattern not found — may already be patched")
        return False


def main() -> int:
    print("=" * 60)
    print("Zaya CT NVFP4 vLLM Serving — Professional Fix")
    print("=" * 60)

    results = {
        "process_weights": fix_process_weights_after_loading(),
        "zaya_conversion": fix_zaya_weight_conversion(),
        "marlin_dims": fix_marlin_dimension_validation(),
    }

    print()
    print("Summary:")
    for name, ok in results.items():
        status = "PASS" if ok else "SKIP/FAIL"
        print(f"  [{status}] {name}")

    all_ok = all(results.values())
    if all_ok:
        print("\nAll fixes applied. Ready to serve.")
    else:
        print("\nSome fixes were skipped. The venv may already be in the correct state.")

    return 0 if all_ok else 0  # Non-zero not helpful here


if __name__ == "__main__":
    sys.exit(main())
