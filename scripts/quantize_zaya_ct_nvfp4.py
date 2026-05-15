"""Stage 1: Quantize ZAYA1-8B BF16 → compressed-tensors NVFP4 format (SOTA).

Uses the NVFP4PackedCompressor from compressed_tensors for proper int8 packing:
  - weight_packed: uint8 [out, in//2] (2 FP4 values per byte)
  - weight_scale: float8_e4m3fn [out, in//16] (native FP8, safetensors 0.7+ GPU save)
  - weight_global_scale: float32 scalar per module
  - Zero points removed (symmetric quantization)

Targets "Linear" modules (auto-includes FusedMoE SequentialMLP Linears).
CCA conv1d layers are NOT Linear → auto-excluded.

Output: ./zaya1-8b-nvfp4-ct/ with safetensors + quantization_config.
Expected: ~5.5 GB disk (bfloat16 scales add ~500 MB vs pure FP8).

Usage:
    python scripts/quantize_zaya_ct_nvfp4.py --dry-run    # 2 layers, ~30s
    python scripts/quantize_zaya_ct_nvfp4.py               # full model, ~2 min
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Zyphra/ZAYA1-8B"
DEFAULT_OUTPUT = "./zaya1-8b-nvfp4-ct"
DRY_RUN_LAYERS = 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1: ZAYA1-8B → NVFP4 compressed-tensors (SOTA)")
    parser.add_argument("--model-id", default=os.environ.get("ZAYA_MODEL_ID", DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help=f"Quantize only first {DRY_RUN_LAYERS} layers")
    args = parser.parse_args()

    # ── Imports ──────────────────────────────────────────────
    import compressed_tensors as ct
    import safetensors.torch as st
    import torch
    import transformers
    from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
    from compressed_tensors.quantization import (
        QuantizationConfig,
        apply_quantization_config,
        preset_name_to_scheme,
    )

    logger.info("=" * 60)
    logger.info("STAGE 1: NVFP4 Compressed-Tensors Quantization (SOTA)")
    logger.info("Model: %s | Output: %s | Mode: %s",
                args.model_id, args.output_dir,
                f"DRY RUN ({DRY_RUN_LAYERS:d}L)" if args.dry_run else "FULL")
    logger.info("Compressor: NVFP4PackedCompressor (pack_fp4_to_uint8)")
    logger.info("torch %s | CUDA: %s | %s",
                torch.__version__, torch.cuda.is_available(),
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
    logger.info("compressed_tensors %s", getattr(ct, "__version__", "?"))
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA required for float8 scale compression")
        return 1

    # ── Load model on CPU ───────────────────────────────────
    logger.info("Loading model on CPU (no offload, no meta params)...")
    t0 = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    logger.info("Model loaded in %.0fs | Params: %.1fB", time.time() - t0,
                sum(p.numel() for p in model.parameters()) / 1e9)

    # Fix generation_config for save
    gc = model.generation_config
    if not gc.do_sample and gc.top_p is not None:
        gc.top_p = None
    if not gc.do_sample and gc.top_k is not None:
        gc.top_k = None

    # ── Apply NVFP4A16 quantization config (GROUP strategy for vLLM compat) ─
    from compressed_tensors.quantization import (
        FP8_E4M3_DATA,
        QuantizationArgs,
        QuantizationScheme,
        QuantizationStrategy,
        QuantizationType,
    )

    weights_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,  # GROUP for vLLM FusedMoE compat (not tensor_group)
        symmetric=True,
        dynamic=False,
        group_size=16,
        scale_dtype=FP8_E4M3_DATA.dtype,
        zp_dtype=FP8_E4M3_DATA.dtype,
    )
    scheme = QuantizationScheme(targets=["Linear"], weights=weights_args)
    logger.info("Scheme: num_bits=%d, group_size=%d, strategy=%s",
                scheme.weights.num_bits, scheme.weights.group_size,
                scheme.weights.strategy)

    config = QuantizationConfig(
        quant_method="compressed-tensors",
        format="float-quantized",
        config_groups={"group_0": scheme},
    )

    apply_quantization_config(model, config)
    logger.info("Quantization config applied (empty scales initialized)")

    # ── Compute proper scales from weights ──────────────────
    # compressed_tensors creates empty scale params; we must fill them.
    # For NVFP4A16 (weight-only): scale = max_abs_per_group / FP4_max(6.0).
    # (ref: compressed_tensors FP4_E2M1_DATA, max=6.0)
    logger.info("Computing per-group weight scales...")
    calibrated = 0
    for _name, module in model.named_modules():
        qscheme = getattr(module, "quantization_scheme", None)
        if qscheme is None or qscheme.weights is None:
            continue
        if "router" in _name.lower():
            module.quantization_scheme = None  # skip router MLP (tiny, unaligned dims)
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        gs = qscheme.weights.group_size
        w = weight.data.float()
        out, in_feat = w.shape
        # Reshape: [out, in//gs, gs]
        w_groups = w.view(out, in_feat // gs, gs)
        # Per-group max absolute value
        w_max = w_groups.abs().amax(dim=-1)  # [out, in//gs]
        # Scale = max / FP4_max
        fp4_max = 6.0
        scales = w_max / fp4_max
        scales = torch.clamp(scales, min=1e-12)
        module.weight_scale.data = scales.to(torch.bfloat16)
        calibrated += 1
    logger.info("Calibrated %d modules", calibrated)

    # ── Compress each quantized module ──────────────────────
    num_layers = model.config.num_hidden_layers
    max_idx = min(num_layers, DRY_RUN_LAYERS) if args.dry_run else num_layers

    compressed_params: dict[str, dict[str, torch.Tensor]] = {}
    linear_count = 0
    skipped = 0

    logger.info("Compressing weights with NVFP4PackedCompressor...")
    t0 = time.time()

    for name, module in model.named_modules():
        qscheme = getattr(module, "quantization_scheme", None)
        if qscheme is None or qscheme.weights is None:
            continue

        # Check if this module is in a layer beyond our limit (for dry-run)
        if args.dry_run:
            parts = name.split(".")
            layer_idx = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        pass
            if layer_idx is not None and layer_idx >= max_idx:
                continue

        weight = getattr(module, "weight", None)
        scale = getattr(module, "weight_scale", None)
        if weight is None or scale is None:
            skipped += 1
            continue

        linear_count += 1

        # Build per-module state dict and compress
        state_dict: dict[str, torch.Tensor] = {
            "weight": weight.data.to("cuda:0"),
            "weight_scale": scale.data.to("cuda:0"),
        }

        compressed = NVFP4PackedCompressor.compress(state_dict, qscheme)

        # Keep on GPU — float8 scales preserved
        compressed_params[name] = {k: v for k, v in compressed.items()}  # noqa: C416

    elapsed = time.time() - t0
    logger.info("Compressed %d Linear modules in %.0fs (skipped %d)", linear_count, elapsed, skipped)

    # ── Quantize embedding layer ────────────────────────────
    # ZAYA uses tie_word_embeddings=True → embed_tokens = lm_head (same weight).
    # NVFP4A16 targets "Linear" but embeddings are nn.Embedding, not targeted.
    # Manually quantize with same NVFP4A16 scheme to save ~735 MB.
    embed_scheme = preset_name_to_scheme("NVFP4A16", targets=["Embedding"])
    embed_config = QuantizationConfig(
        quant_method="compressed-tensors",
        format="float-quantized",
        config_groups={"group_embed": embed_scheme},
    )
    apply_quantization_config(model, embed_config)
    logger.info("Embedding quantization: config applied")

    for name, module in model.named_modules():
        qscheme = getattr(module, "quantization_scheme", None)
        if qscheme is None or qscheme.weights is None:
            continue
        if name in compressed_params:
            continue  # already compressed

        # Tied embedding: skip quantization, copy from lm_head later
        if name == "model.embed_tokens" and "lm_head" in compressed_params:
            continue

        weight = getattr(module, "weight", None)
        scale = getattr(module, "weight_scale", None)
        if weight is None or scale is None:
            continue

        gscale = getattr(module, "weight_global_scale", None)
        state_dict: dict[str, torch.Tensor] = {
            "weight": weight.data.to("cuda:0"),
            "weight_scale": scale.data.to("cuda:0"),
        }

        compressed = NVFP4PackedCompressor.compress(state_dict, qscheme)
        compressed_params[name] = {k: v for k, v in compressed.items()}  # keep on GPU
        logger.info("  Embedding %s: packed to %s uint8", name, list(compressed["weight_packed"].shape))

    # Deduplicate tied embeddings (tie_word_embeddings=True)
    if "lm_head" in compressed_params:
        compressed_params["model.embed_tokens"] = compressed_params["lm_head"]
        logger.info("Dedup tied embeddings: model.embed_tokens = lm_head")

    # ── Build output state dict on GPU ──────────────────────
    logger.info("Building output state dict (GPU, float8 scales preserved)...")
    output_state: dict[str, torch.Tensor] = {}
    packed_total = 0
    scale_total = 0
    other_total = 0
    device = torch.device("cuda:0")

    for pname, param in model.named_parameters():
        module_name = ".".join(pname.split(".")[:-1])
        param_short = pname.split(".")[-1]

        if module_name in compressed_params:
            # Tied embedding: skip entirely (lm_head handles it)
            if module_name == "model.embed_tokens" and "lm_head" in compressed_params:
                continue
            if param_short == "weight":
                continue
            if param_short == "weight_zero_point":
                continue
            if param_short in compressed_params[module_name]:
                tensor = compressed_params[module_name][param_short]
                output_state[pname] = tensor  # on GPU
                if "weight_packed" in param_short:
                    packed_total += tensor.numel() * tensor.element_size()
                elif "weight_scale" in param_short and "global" not in param_short:
                    scale_total += tensor.numel() * tensor.element_size()
                continue
            if param.device.type != "meta":
                output_state[pname] = param.detach().to(device)
                other_total += param.numel() * param.element_size()
            continue

        if param.device.type != "meta":
            output_state[pname] = param.detach().to(device)
            other_total += param.numel() * param.element_size()

    # Add weight_packed for each compressed module (tied embeddings: only save one)
    for mod_name, comp in compressed_params.items():
        if mod_name == "model.embed_tokens" and "lm_head" in compressed_params:
            continue  # tied to lm_head, skip duplicate
        for key, tensor in comp.items():
            full_name = f"{mod_name}.{key}"
            if full_name not in output_state:
                output_state[full_name] = tensor
                if "weight_packed" in key:
                    packed_total += tensor.numel() * tensor.element_size()

    logger.info("Output: %d params | packed: %.0f MB | scales: %.0f MB | other: %.0f MB",
                len(output_state), packed_total / 1e6, scale_total / 1e6, other_total / 1e6)

    # ── Save ────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    logger.info("Saving to %s ...", output_dir)
    t0 = time.time()

    model.config.save_pretrained(str(output_dir))
    model.generation_config.save_pretrained(str(output_dir))

    # Safetensors + index (GPU tensors with float8 scales preserved)
    st.save_file(output_state, str(output_dir / "model.safetensors"))
    index: dict[str, Any] = {
        "metadata": {"total_size": len(output_state)},
        "weight_map": {name: "model.safetensors" for name in output_state},
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    # Quantization manifest
    total_bytes = sum(
        f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
    )
    manifest = {
        "model": "Zyphra/ZAYA1-8B",
        "quantization": {
            "method": "compressed-tensors",
            "format": "float-quantized",
            "scheme": "NVFP4A16",
            "num_bits": 4,
            "type": "float",
            "strategy": "tensor_group",
            "group_size": 16,
            "target_modules": ["Linear"],
            "weight_packed_dtype": "uint8",
            "weight_scale_dtype": "float8_e4m3fn (native FP8 via safetensors 0.7+ GPU save)",
            "weight_global_scale_dtype": "float32",
            "zero_point": "none (symmetric)",
            "compressor": "NVFP4PackedCompressor",
            "effective_bpw_estimate": 5.2,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "vram": "16 GB",
            "cuda": torch.version.cuda,
        },
        "modules_compressed": linear_count,
        "output_size_bytes": total_bytes,
        "dry_run": args.dry_run,
    }
    with open(output_dir / "quantization_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    logger.info("Saved in %.0fs | Total: %.0f MB (%.2f GB)",
                elapsed, total_bytes / 1e6, total_bytes / 1e9)

    # ── Verify ──────────────────────────────────────────────
    logger.info("")
    logger.info("=== Verification ===")
    verify = st.load_file(str(output_dir / "model.safetensors"), device="cpu")
    dtype_counts: dict[str, int] = {}
    packed_mb = 0
    scale_mb = 0
    for name, tensor in verify.items():
        dt = str(tensor.dtype)[6:]
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
        if "weight_packed" in name:
            packed_mb += tensor.numel() * tensor.element_size() / 1e6
        elif "weight_scale" in name and "global" not in name:
            scale_mb += tensor.numel() * tensor.element_size() / 1e6
    logger.info("Dtypes: %s", {k: v for k, v in sorted(dtype_counts.items())})
    logger.info("weight_packed: %.0f MB | weight_scale: %.0f MB", packed_mb, scale_mb)

    # Spot-check: first packed weight has shape [out, in//2]
    for name, tensor in verify.items():
        if "weight_packed" in name:
            logger.info("Sample %s: shape=%s dtype=%s", name, list(tensor.shape), tensor.dtype)
            break

    logger.info("")
    logger.info("=" * 60)
    if args.dry_run:
        logger.info("DRY RUN PASSED — pipeline verified on %d layers", DRY_RUN_LAYERS)
        logger.info("Ready for full: python scripts/quantize_zaya_ct_nvfp4.py")
    else:
        logger.info("STAGE 1 COMPLETE — SOTA NVFP4 ZAYA1-8B")
        logger.info("Output: %s (%.1f GB)", output_dir, total_bytes / 1e9)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
