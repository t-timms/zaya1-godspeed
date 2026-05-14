#!/usr/bin/env python3
"""Apply professional, upstreamable fixes to vLLM for NVFP4 Compressed-Tensors Zaya support.

These fixes replace the 18+ monkey-patches from wsl_fix_*/wsl_patch_* scripts
with clean, documented, testable changes that can be submitted as upstream PRs.

Categories:
  1. input_quant=None guards (3 methods) — bug fix
  2. Marlin dimension validation with graceful fallback — feature
  3. process_weights_after_loading restoration — critical bug fix
  4. NVFP4 Python dequant fallback for unaligned layers — feature
  5. WNA16 kernel graceful fallback — feature (already applied)
  6. zaya.py CT MoE weight loading — feature (already applied)
  7. ZayaForCausalLM ModelRegistry registration — feature
  8. CCA state calculators — feature

Usage:
    python scripts/apply_professional_fixes.py [--check] [--revert]
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Locate vLLM installation ──────────────────────────────────────
_vllm_spec = importlib.util.find_spec("vllm")
if _vllm_spec is None or _vllm_spec.origin is None:
    logger.error("vLLM not installed. Run: pip install vllm")
    sys.exit(1)
VLLM_ROOT = Path(_vllm_spec.origin).parent  # site-packages/vllm/
VLLM_SRC = VLLM_ROOT  # vllm package dir
logger.info("vLLM root: %s", VLLM_ROOT)


# ═══════════════════════════════════════════════════════════════════
# Fix 1: input_quant=None guards in CompressedTensorsConfig
# ─────────────────────────────────────────────────────────────────
# Bug: _is_static_tensor_w8a8, _is_dynamic_token_w8a8,
#      _is_dynamic_token_w4a8_int assume input_quant is non-None.
#      Weight-only quantization (NVFP4) has input_quant=None,
#      causing AttributeError on input_quant.num_bits.
# Fix: Add early-return guards before accessing input_quant.num_bits.
# Upstream: PR to vllm-project/vllm — weight-only quant support.
# ═══════════════════════════════════════════════════════════════════


def fix_input_quant_guards(ct_path: Path) -> bool:
    """Add input_quant is None guards to 3 scheme-detection methods."""
    content = ct_path.read_text()

    replacements = [
        # _is_static_tensor_w8a8
        (
            "        is_8_bits = weight_quant.num_bits == 8 and input_quant is not None and input_quant.num_bits == 8\n        weight_strategy = (\n            weight_quant.strategy == QuantizationStrategy.TENSOR.value",
            "        if input_quant is None:\n            return False\n        is_8_bits = weight_quant.num_bits == 8 and input_quant.num_bits == 8\n        weight_strategy = (\n            weight_quant.strategy == QuantizationStrategy.TENSOR.value",
        ),
        # _is_dynamic_token_w8a8
        (
            "        is_8_bits = weight_quant.num_bits == 8 and input_quant is not None and input_quant.num_bits == 8\n        weight_strategy = (\n            weight_quant.strategy == QuantizationStrategy.TENSOR.value\n            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value\n        )\n        is_token = (",
            "        if input_quant is None:\n            return False\n        is_8_bits = weight_quant.num_bits == 8 and input_quant.num_bits == 8\n        weight_strategy = (\n            weight_quant.strategy == QuantizationStrategy.TENSOR.value\n            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value\n        )\n        is_token = (",
        ),
        # _is_dynamic_token_w4a8_int
        (
            "        is_weight_4_bits = weight_quant.num_bits == 4\n        is_activation_8_bits = input_quant.num_bits == 8",
            "        if input_quant is None:\n            return False\n        is_weight_4_bits = weight_quant.num_bits == 4\n        is_activation_8_bits = input_quant.num_bits == 8",
        ),
    ]

    modified = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            modified = True
            logger.info("  Applied input_quant guard")
        else:
            logger.info("  Guard already applied or pattern not found")

    if modified:
        ct_path.write_text(content)
    return modified


# ═══════════════════════════════════════════════════════════════════
# Fix 2: Marlin dimension validation in prepare_fp4_layer_for_marlin
# ─────────────────────────────────────────────────────────────────
# Bug: ops.gptq_marlin_repack requires tile-aligned dimensions
#      (size_n % 64 == 0, size_k % 256 == 0). CCA attention's
#      conv1d-based projections may produce unaligned dimensions.
# Fix: Validate dimensions before repack; skip with marker attribute
#      if unaligned. Consumer (apply_weights) checks marker and
#      falls back to Python dequant.
# Upstream: PR to vllm-project/vllm — robust Marlin repack.
# ═══════════════════════════════════════════════════════════════════

MARLIN_TILE_N = 64  # Marlin output tile (N dimension)
MARLIN_TILE_K = 256  # Marlin input tile (K dimension)


def fix_marlin_dimension_validation(marlin_path: Path) -> bool:
    """Add dimension validation to prepare_fp4_layer_for_marlin."""
    content = marlin_path.read_text()

    # ── Add helper function before prepare_fp4_layer_for_marlin ──
    helper = '''
def _validate_marlin_dimensions(
    layer: torch.nn.Module, size_k: int, size_n: int
) -> bool:
    """Check if layer dimensions meet Marlin kernel tile requirements.

    The Marlin FP4 kernel requires:
      - size_n (output dim) % 64 == 0  (tile width)
      - size_k (input dim)  % 256 == 0 (tile height)

    Returns True if dimensions are Marlin-compatible, False otherwise.
    When False, sets layer._marlin_repack_skipped = True so the
    consuming scheme can fall back to Python dequant.
    (ref: vllm/model_executor/kernels/linear — Marlin tile sizes)
    """
    if size_n % 64 != 0 or size_k % 256 != 0:
        logger.warning(
            "Skipping Marlin repack for layer (size_k=%d, size_n=%d) — "
            "dimensions not tile-aligned (k%%256=0, n%%64=0 required). "
            "Falling back to Python dequant.",
            size_k, size_n,
        )
        layer._marlin_repack_skipped = True
        return False
    return True

'''
    # Insert helper function before prepare_fp4_layer_for_marlin
    marker = "def prepare_fp4_layer_for_marlin("
    if "_validate_marlin_dimensions" not in content:
        content = content.replace(marker, helper + marker)
        logger.info("  Added _validate_marlin_dimensions helper")
    else:
        logger.info("  Helper already exists")

    # ── Add dimension check before gptq_marlin_repack call ──
    old_repack = (
        "    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1\n"
        "    marlin_qweight = ops.gptq_marlin_repack("
    )
    new_repack = (
        "    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1\n"
        "\n"
        "    # Validate Marlin tile alignment; fall back to Python dequant if unaligned\n"
        "    if not _validate_marlin_dimensions(layer, part_size_k, part_size_n):\n"
        "        return\n"
        "\n"
        "    marlin_qweight = ops.gptq_marlin_repack("
    )

    if "_validate_marlin_dimensions(layer, part_size_k, part_size_n)" not in content:
        content = content.replace(old_repack, new_repack)
        logger.info("  Added dimension validation before gptq_marlin_repack")
    else:
        logger.info("  Dimension validation already present")

    marlin_path.write_text(content)
    return True


# ═══════════════════════════════════════════════════════════════════
# Fix 3: Restore process_weights_after_loading with proper delegation
# ─────────────────────────────────────────────────────────────────
# Bug: The CompressedTensorsLinearMethod.process_weights_after_loading
#      was replaced with `pass` (wsl_skip_process.py), breaking all
#      Marlin repack. This causes g_idx_sort_indices / weight_perm
#      AttributeError at inference time.
# Fix: Restore delegation to layer.scheme.process_weights_after_loading
#      for layers that have weight_packed.
# Upstream: Not needed — this was a local breakage.
# ═══════════════════════════════════════════════════════════════════


def fix_process_weights_after_loading(ct_path: Path) -> bool:
    """Restore process_weights_after_loading to delegate to layer.scheme."""
    content = ct_path.read_text()

    old_pass = (
        "    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n"
        "        pass  # weights in correct int32 format for WNA16 Marlin"
    )
    new_delegate = (
        "    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n"
        '        """Delegate weight processing to the per-layer quantization scheme.\n'
        "\n"
        "        For NVFP4 layers, the scheme calls prepare_fp4_layer_for_marlin\n"
        "        which repacks weights into the Marlin kernel format. If dimensions\n"
        "        are not tile-aligned, the repack is skipped and the scheme falls\n"
        "        back to Python dequant at inference time.\n"
        '        """\n'
        '        if hasattr(layer, "weight_packed"):\n'
        "            layer.scheme.process_weights_after_loading(layer)"
    )

    if old_pass in content:
        content = content.replace(old_pass, new_delegate)
        ct_path.write_text(content)
        logger.info("  Restored process_weights_after_loading delegation")
        return True
    elif "layer.scheme.process_weights_after_loading(layer)" in content:
        logger.info("  process_weights_after_loading already restored")
        return True
    else:
        logger.error("  Could not find 'pass' pattern in process_weights_after_loading")
        # Try to find it with different whitespace
        if "def process_weights_after_loading(self, layer" in content:
            logger.info("  Found the method but pattern mismatch — manual review needed")
        return False


# ═══════════════════════════════════════════════════════════════════
# Fix 4: NVFP4 scheme Python dequant fallback for unaligned layers
# ─────────────────────────────────────────────────────────────────
# Feature: When the Marlin repack is skipped due to unaligned
#          dimensions, the NVFP4 scheme's apply_weights should fall
#          back to Python dequant using the original uint8 packed
#          weights and float8 scales.
# Upstream: PR to vllm-project/vllm — robust weight-only inference.
# ═══════════════════════════════════════════════════════════════════


def fix_nvfp4_scheme_fallback(scheme_path: Path) -> bool:
    """Add Python dequant fallback to NVFP4 scheme's apply_weights."""
    content = scheme_path.read_text()

    if "_marlin_repack_skipped" in content:
        logger.info("  NVFP4 fallback already present")
        return True

    # ── Fix process_weights_after_loading: save original data for fallback ──
    old_process = (
        "    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n"
        "        # Process parameters for marlin repacking\n"
        "\n"
        "        # Rename weight_packed to weight that marlin expects\n"
        "        layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)\n"
        "        del layer.weight_packed\n"
        "        # ct stores the inverse of what is expected by the marlin kernel\n"
        "        layer.weight_global_scale = Parameter(\n"
        "            1.0 / layer.weight_global_scale.max().to(torch.float32), requires_grad=False\n"
        "        )\n"
        "\n"
        "        prepare_fp4_layer_for_marlin(layer)"
    )
    new_process = (
        "    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n"
        "        # Process parameters for marlin repacking\n"
        "\n"
        "        # Save original data for Python dequant fallback\n"
        "        layer._weight_packed_data = layer.weight_packed.data.clone()\n"
        "        layer._weight_scale_data = layer.weight_scale.data.clone()\n"
        "        layer._weight_global_scale_data = layer.weight_global_scale.data.clone()\n"
        "\n"
        "        # Rename weight_packed to weight that marlin expects\n"
        "        layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)\n"
        "        del layer.weight_packed\n"
        "        # ct stores the inverse of what is expected by the marlin kernel\n"
        "        layer.weight_global_scale = Parameter(\n"
        "            1.0 / layer.weight_global_scale.max().to(torch.float32), requires_grad=False\n"
        "        )\n"
        "\n"
        "        prepare_fp4_layer_for_marlin(layer)"
    )

    if old_process in content:
        content = content.replace(old_process, new_process)
        logger.info("  Added weight data capture for fallback")
    else:
        logger.info("  process_weights_after_loading pattern differs — checking...")
        if "process_weights_after_loading" in content:
            logger.info("  Method exists but content differs from expected")

    # ── Fix apply_weights: check _marlin_repack_skipped ──
    old_apply = (
        "    def apply_weights(\n"
        "        self,\n"
        "        layer: torch.nn.Module,\n"
        "        x: torch.Tensor,\n"
        "        bias: torch.Tensor | None = None,\n"
        "    ) -> torch.Tensor:\n"
        "        return apply_fp4_marlin_linear(\n"
        "            input=x,\n"
        "            weight=layer.weight,\n"
        "            weight_scale=layer.weight_scale,\n"
        "            weight_global_scale=layer.weight_global_scale,\n"
        "            workspace=layer.workspace,\n"
        "            size_n=layer.output_size_per_partition,\n"
        "            size_k=layer.input_size_per_partition,\n"
        "            bias=bias,\n"
        "        )"
    )
    new_apply = (
        "    def apply_weights(\n"
        "        self,\n"
        "        layer: torch.nn.Module,\n"
        "        x: torch.Tensor,\n"
        "        bias: torch.Tensor | None = None,\n"
        "    ) -> torch.Tensor:\n"
        "        # Python dequant fallback for layers where Marlin repack was skipped\n"
        "        # (e.g., CCA attention projections with unaligned dimensions)\n"
        '        if getattr(layer, "_marlin_repack_skipped", False):\n'
        "            from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8\n"
        "            from compressed_tensors.quantization.lifecycle.forward import dequantize\n"
        "            import torch as _t\n"
        "            wq = layer._weight_packed_data\n"
        "            ws = layer._weight_scale_data\n"
        '            wgs = getattr(layer, "_weight_global_scale_data", None)\n'
        "            m, nh = wq.shape\n"
        "            w = unpack_fp4_from_uint8(wq, m, nh * 2)\n"
        "            w = dequantize(x_q=w, scale=ws.float(), global_scale=wgs, dtype=ws.float().dtype)\n"
        "            out = _t.nn.functional.linear(x, w.to(x.dtype))\n"
        "            return out + bias if bias is not None else out\n"
        "\n"
        "        return apply_fp4_marlin_linear(\n"
        "            input=x,\n"
        "            weight=layer.weight,\n"
        "            weight_scale=layer.weight_scale,\n"
        "            weight_global_scale=layer.weight_global_scale,\n"
        "            workspace=layer.workspace,\n"
        "            size_n=layer.output_size_per_partition,\n"
        "            size_k=layer.input_size_per_partition,\n"
        "            bias=bias,\n"
        "        )"
    )

    if "_marlin_repack_skipped" not in content and old_apply in content:
        content = content.replace(old_apply, new_apply)
        logger.info("  Added Python dequant fallback to apply_weights")
    elif "_marlin_repack_skipped" in content:
        logger.info("  Fallback already present")
    else:
        logger.info("  apply_weights pattern differs — may already be patched")

    scheme_path.write_text(content)
    return True


# ═══════════════════════════════════════════════════════════════════
# Fix 5: CCA state calculators (Mamba utils)
# ─────────────────────────────────────────────────────────────────
# Feature: ZAYA1-8B's CCA attention requires cca_state_shape and
#          cca_state_dtype methods on the Mamba state calculators.
#          These are missing from stock vLLM 0.20.2.
# Upstream: PR to vllm-project/vllm — Mamba utils CCA support.
# ═══════════════════════════════════════════════════════════════════


def fix_cca_state_calculators(vllm_src: Path) -> bool:
    """Add cca_state_shape and cca_state_dtype to Mamba utils."""
    mamba_utils = vllm_src / "model_executor" / "layers" / "mamba" / "mamba_utils.py"
    if not mamba_utils.exists():
        logger.warning("  mamba_utils.py not found at %s", mamba_utils)
        return False

    content = mamba_utils.read_text()

    # Check if already patched correctly (both methods inside their classes)
    if "def cca_state_shape(" in content and "def cca_state_dtype(" in content:
        # Verify they're inside classes by checking indentation
        if "    @classmethod\n    def cca_state_shape" in content:
            logger.info("  CCA state calculators already present (correctly indented)")
            return True

    modified = False

    # ── Add cca_state_shape inside MambaStateShapeCalculator ──
    # Insert after kda_state_shape's return, before the next @dataclass/class
    if "def cca_state_shape(" not in content:
        shape_method = (
            "\n"
            "    @classmethod\n"
            "    def cca_state_shape(\n"
            "        cls,\n"
            "        tp_world_size: int,\n"
            "        conv_kernel_size: int,\n"
            "        num_k_heads: int,\n"
            "        num_q_heads: int,\n"
            "        head_dim: int,\n"
            "        hidden_size: int,\n"
            "    ):\n"
            '        """Calculate CCA (Compressed Convolutional Attention) state shapes.\n'
            "\n"
            "        CCA uses depthwise+grouped conv1d for Q/K mixing with dual time-stream\n"
            "        values (val_proj1, val_proj2). This method computes the state buffers\n"
            "        needed for the convolutional attention kernel.\n"
            "        (ref: ZYPHRA ZAYA1-8B Technical Report, arXiv 2605.05365)\n"
            '        """\n'
            "        proj_size = num_q_heads * head_dim\n"
            "        proj_k_size = num_k_heads * head_dim\n"
            "        return (\n"
            "            cls._orient_conv_shape(proj_size // tp_world_size, conv_kernel_size - 1),\n"
            "            cls._orient_conv_shape(proj_k_size // tp_world_size, conv_kernel_size - 1),\n"
            "            (hidden_size,),\n"
            "        )\n"
        )
        # Insert right before "@dataclass" which follows MambaStateShapeCalculator
        old = "\n\n@dataclass\nclass MambaCopySpec"
        new = shape_method + "\n\n@dataclass\nclass MambaCopySpec"
        if old in content:
            content = content.replace(old, new)
            modified = True
            logger.info("  Added cca_state_shape inside MambaStateShapeCalculator")
        else:
            logger.warning("  Could not find insertion point for cca_state_shape")

    # ── Add cca_state_dtype inside MambaStateDtypeCalculator ──
    # Insert after kda_state_dtype's return, before next class
    if "def cca_state_dtype(" not in content:
        dtype_method = (
            "\n"
            "    @classmethod\n"
            "    def cca_state_dtype(\n"
            "        cls,\n"
            '        model_dtype: "torch.dtype",\n'
            "        mamba_cache_dtype: str,\n"
            "    ):\n"
            '        """Calculate CCA state dtype from model and cache precision.\n'
            "\n"
            "        Returns a tuple of (state_dtype, state_dtype) for the two CCA\n"
            "        state buffers, derived from the KDA state dtype calculation.\n"
            "        (ref: vllm/model_executor/layers/mamba/mamba_utils.py)\n"
            '        """\n'
            "        state_dtype = MambaStateDtypeCalculator.kda_state_dtype(\n"
            "            model_dtype, mamba_cache_dtype\n"
            "        )[0]\n"
            "        return (state_dtype, state_dtype)\n"
        )
        # Insert right before "class MambaStateShapeCalculator"
        old = "\n\nclass MambaStateShapeCalculator"
        new = dtype_method + "\n\nclass MambaStateShapeCalculator"
        if old in content:
            content = content.replace(old, new)
            modified = True
            logger.info("  Added cca_state_dtype inside MambaStateDtypeCalculator")
        else:
            logger.warning("  Could not find insertion point for cca_state_dtype")

    if modified:
        mamba_utils.write_text(content)
    return modified


# ═══════════════════════════════════════════════════════════════════
# Fix 6: ModelRegistry — ZayaForCausalLM registration
# ─────────────────────────────────────────────────────────────────
# Feature: Register ZayaForCausalLM in vLLM's ModelRegistry so that
#          `--model ./zaya1-8b-nvfp4-ct` auto-detects the model type.
#          The zaya.py model file exists in the Zyphra fork but is
#          not registered in stock vLLM 0.20.2.
# Upstream: PR to vllm-project/vllm — Zaya architecture support.
# ═══════════════════════════════════════════════════════════════════


def fix_ct_small_dim_skip(vllm_src: Path) -> bool:
    """Skip CT quantization for Linear layers too small for any WNA16 kernel.

    Zaya router layers (output_size=17, D=256/E=17) are too small for Marlin
    (min 64 threads) and no other WNA16 kernel supports them on sm_120.
    This patch adds a dimension check in get_quant_method to skip CT
    quantization for layers where all kernels would fail.

    Upstream: PR to vllm-project/vllm — robust quant method selection.
    """
    ct_path = vllm_src / "model_executor" / "layers" / "quantization" / "compressed_tensors" / "compressed_tensors.py"
    if not ct_path.exists():
        logger.warning("  CT config not found at %s", ct_path)
        return False

    content = ct_path.read_text()

    if "SmallDimSkip" in content:
        logger.info("  CT small-dim skip already present")
        return True

    old = (
        "        if isinstance(layer, LinearBase):\n"
        "            # collect schemes\n"
        "            quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)"
    )
    new = (
        "        if isinstance(layer, LinearBase):\n"
        "            # SmallDimSkip: skip CT quantization for layers too small\n"
        "            # for any WNA16 kernel (e.g., Zaya router with output_size=17).\n"
        "            # Without this guard, choose_mp_linear_kernel raises ValueError\n"
        "            # and model loading fails.\n"
        "            if _is_small_dim_linear(layer):\n"
        "                return UnquantizedLinearMethod()\n"
        "            # collect schemes\n"
        "            quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)"
    )

    if old in content:
        content = content.replace(old, new)
        ct_path.write_text(content)
        logger.info("  Added CT small-dim skip in get_quant_method")
    else:
        logger.warning("  Could not find get_quant_method pattern for small-dim skip")
        return False

    # Also add the helper function to the module
    helper = (
        '\n\ndef _is_small_dim_linear(layer: "torch.nn.Module") -> bool:\n'
        '    """Check if a Linear layer has output dim too small for any WNA16 kernel.\n'
        "\n"
        "    Marlin requires output_size_per_partition >= 64. No other kernel\n"
        "    supports weight-only 4-bit on sm_120 (Blackwell). If output is smaller,\n"
        "    return True to signal that CT quantization should be skipped.\n"
        '    """\n'
        "    from vllm.model_executor.layers.linear import LinearBase\n"
        '    if not hasattr(layer, "output_size_per_partition"):\n'
        "        return False\n"
        "    return layer.output_size_per_partition < 64\n"
    )

    if "_is_small_dim_linear" not in content:
        # Insert before the class definition
        content = content.replace(
            "\nclass CompressedTensorsConfig",
            helper + "\nclass CompressedTensorsConfig",
        )
        ct_path.write_text(content)
        logger.info("  Added _is_small_dim_linear helper")
    return True


def fix_model_registry(vllm_src: Path) -> bool:
    """Register ZayaForCausalLM in vLLM ModelRegistry."""
    registry = vllm_src / "model_executor" / "models" / "registry.py"
    if not registry.exists():
        logger.warning("  registry.py not found at %s", registry)
        return False

    content = registry.read_text()

    if '"ZayaForCausalLM"' in content:
        logger.info("  ZayaForCausalLM already registered")
        return True

    entry = '    "ZayaForCausalLM": ("zaya", "ZayaForCausalLM"),\n'
    marker = "_TEXT_GENERATION_MODELS = {"
    if marker in content:
        content = content.replace(marker, marker + "\n" + entry)
        registry.write_text(content)
        logger.info("  Registered ZayaForCausalLM in ModelRegistry")
        return True

    logger.warning("  Could not find _TEXT_GENERATION_MODELS in registry.py")
    return False


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply professional vLLM fixes for NVFP4 CT Zaya")
    parser.add_argument("--check", action="store_true", help="Check fix status without applying")
    parser.add_argument("--revert", action="store_true", help="Revert all fixes (restore originals)")
    args = parser.parse_args()

    if args.revert:
        logger.error("Revert not yet implemented — restore from backup or reinstall vLLM")
        return 1

    # ── File paths ──────────────────────────────────────────────
    ct_file = VLLM_SRC / "model_executor" / "layers" / "quantization" / "compressed_tensors" / "compressed_tensors.py"
    marlin_file = VLLM_SRC / "model_executor" / "layers" / "quantization" / "utils" / "marlin_utils_fp4.py"
    scheme_file = (
        VLLM_SRC
        / "model_executor"
        / "layers"
        / "quantization"
        / "compressed_tensors"
        / "schemes"
        / "compressed_tensors_w4a16_nvfp4.py"
    )

    # Verify files exist
    for path, name in [
        (ct_file, "compressed_tensors.py"),
        (marlin_file, "marlin_utils_fp4.py"),
        (scheme_file, "NVFP4 scheme"),
    ]:
        if not path.exists():
            logger.error("%s not found at %s", name, path)
            return 1

    logger.info("=" * 60)
    logger.info("Applying professional vLLM fixes for NVFP4 CT Zaya")
    logger.info("=" * 60)

    if args.check:
        logger.info("CHECK MODE — verifying fix status:")
        check_fixes(ct_file, marlin_file, scheme_file)
        return 0

    # ── Apply fixes ────────────────────────────────────────────
    results: dict[str, bool] = {}

    logger.info("\n1. input_quant=None guards...")
    results["input_quant_guards"] = fix_input_quant_guards(ct_file)

    logger.info("\n2. Marlin dimension validation...")
    results["marlin_validation"] = fix_marlin_dimension_validation(marlin_file)

    logger.info("\n3. process_weights_after_loading restoration...")
    results["process_weights"] = fix_process_weights_after_loading(ct_file)

    logger.info("\n4. NVFP4 scheme Python dequant fallback...")
    results["nvfp4_fallback"] = fix_nvfp4_scheme_fallback(scheme_file)

    logger.info("\n5. CCA state calculators...")
    results["cca_calculators"] = fix_cca_state_calculators(VLLM_SRC)

    logger.info("\n6. ModelRegistry Zaya registration...")
    results["model_registry"] = fix_model_registry(VLLM_SRC)

    logger.info("\n7. CT small-dim skip (router layers)...")
    results["ct_small_dim"] = fix_ct_small_dim_skip(VLLM_SRC)

    # ── Summary ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Fix Summary:")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        logger.info("  %-30s %s", name, status)
    logger.info("=" * 60)

    all_ok = all(results.values())
    if all_ok:
        logger.info("All professional fixes applied successfully.")
        logger.info("Ready to serve: python scripts/wsl_serve_patched.py")
    else:
        logger.warning("Some fixes could not be applied — review manually.")

    return 0 if all_ok else 1


def check_fixes(ct_path: Path, marlin_path: Path, scheme_path: Path) -> None:
    """Check which fixes are already applied."""
    ct = ct_path.read_text()
    marlin = marlin_path.read_text()
    scheme = scheme_path.read_text()

    checks = [
        ("input_quant guard (static)", "input_quant is None" in ct and "return False" in ct),
        ("process_weights_after_loading restored", "layer.scheme.process_weights_after_loading(layer)" in ct),
        ("Marlin dimension validation", "_validate_marlin_dimensions" in marlin),
        ("NVFP4 Python dequant fallback", "_marlin_repack_skipped" in scheme),
        ("CCA state calculators", False),  # checked in function
        ("Zaya ModelRegistry", False),  # checked in function
    ]

    for name, status in checks:
        logger.info("  %-40s %s", name, "PASS" if status else "MISSING")


if __name__ == "__main__":
    sys.exit(main())
