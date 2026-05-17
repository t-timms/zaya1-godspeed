"""Stage 1: Quantize ZAYA1-8B BF16 → compressed-tensors NVFP4 format.

Two schemes supported via --scheme:

  w4a16 (default, original SOTA): NVFP4A16 — 4-bit weights, 16-bit activations.
    Targets "Linear", routers nullified post-apply. Uses NVFP4PackedCompressor.
    Output: ./zaya1-8b-nvfp4-ct/ (~5.5 GB). Run time: ~2 min on CPU. Compatible
    with vLLM's Marlin and CUTLASS W4A16 dispatch paths.

  w4a4 (Path B Phase 1): NVFP4 — 4-bit weights AND 4-bit activations. Adds
    static per-tensor input_global_scale computed from a calibration corpus.
    ignore list (set on QuantizationConfig, not post-hoc): lm_head, router,
    norm, cca, mamba, conv1d — those layers stay BF16. Activation calibration
    runs the BF16 model forward over calibration_data.pt via device_map="auto"
    offload, observing max |activation| per quantized Linear via pre-forward
    hooks, then sets input_global_scale = max / FP4_E2M1_max (6.0).
    Output: ./zaya1-8b-nvfp4-w4a4/ (~5.5 GB + per-Linear fp32 scales).
    Run time: ~1-4 GPU-hours (offload-bound). Requires data/calibration/
    calibration_data.pt from build_calibration_data.py. Compatible with vLLM's
    CutlassNvFp4LinearKernel (force via VLLM_NVFP4_GEMM_BACKEND=cutlass).

Both schemes use NVFP4PackedCompressor:
  - weight_packed: uint8 [out, in//2] (2 FP4 values per byte)
  - weight_scale: float8_e4m3fn [out, in//16]
  - weight_global_scale: float32 scalar per module
  - (w4a4 only) input_global_scale: float32 scalar per fused Linear group
  - Zero points removed (symmetric quantization)

Targets "Linear" (auto-includes FusedMoE SequentialMLP Linears).
CCA conv1d layers are NOT Linear → auto-excluded; for w4a4 they're also
explicit in the ignore regex.

Usage:
    # W4A16 (legacy SOTA path, default)
    python scripts/quantize_zaya_ct_nvfp4.py --dry-run            # 2 layers, ~30s
    python scripts/quantize_zaya_ct_nvfp4.py                       # full model, ~2 min

    # W4A4 (Phase 1 Path B — true NVFP4 for SM120 CUTLASS)
    python scripts/quantize_zaya_ct_nvfp4.py --scheme w4a4 --dry-run   # 4 layers + tiny calibration
    python scripts/quantize_zaya_ct_nvfp4.py --scheme w4a4              # full, 1-4 GPU-hr
    python scripts/quantize_zaya_ct_nvfp4.py --scheme w4a4 \\
        --calibration-data data/calibration/calibration_data.pt \\
        --calibration-num-samples 256                                   # quick iteration
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
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
DEFAULT_OUTPUT_W4A4 = "./zaya1-8b-nvfp4-w4a4"
DEFAULT_CALIBRATION_DATA = "data/calibration/calibration_data.pt"
DRY_RUN_LAYERS = 2
DRY_RUN_LAYERS_W4A4 = 4  # need a few more for hook coverage to make sense
FP4_E2M1_MAX = 6.0  # max magnitude representable in FP4 E2M1 — divisor for scales

# W4A4 layers that MUST stay BF16 (regex patterns for compressed_tensors
# ignore list). compressed_tensors matches against module *paths* from
# named_modules(), not class names — so the CCA pattern must target the
# `qkv` attribute that ZayaAttention uses for its CCA submodule (paths look
# like `model.layers.0.self_attn.qkv.linear_q`). The earlier `re:.*cca.*`
# matched nothing and silently quantized 160 CCA projections to W4A4.
W4A4_IGNORE_PATTERNS = [
    "lm_head",            # tied to embed_tokens, BF16
    "re:.*router.*",      # MoE router (size_n=17, doesn't fit FP4 grid cleanly)
    "re:.*norm.*",        # RMSNorm
    "re:.*qkv.*",         # CCA Q/K/V projections (4 Linears × 40 ATT layers = 160)
]


def _is_quantized_linear(module: Any) -> bool:
    """True iff the module has a quantization_scheme with both weight + input quant."""
    import torch
    qs = getattr(module, "quantization_scheme", None)
    if qs is None:
        return False
    if not isinstance(module, torch.nn.Linear):
        return False
    if getattr(qs, "input_activations", None) is None:
        return False
    return True


def calibrate_input_global_scales(
    model: Any,
    calibration_tensor: Any,
    batch_size: int = 1,
    num_samples: int | None = None,
    log_every: int = 20,
) -> dict[str, float]:
    """Run forward hooks to observe max |activation| per quantized Linear, then
    set input_global_scale = max_act / FP4_E2M1_MAX on each module.

    Returns the raw max-activation dict for diagnostics. Model must have
    quantization_scheme attached to each target Linear (run apply_quantization_config
    first). Hooks are removed before return.

    Memory: model must be on GPU or device_map="auto"-offloaded. Forward batch
    size 1 because 8B BF16 + 1024-token activations near saturates 16 GB.
    """
    import torch

    activation_max: dict[str, float] = {}

    target_modules: list[tuple[str, Any]] = [
        (name, mod) for name, mod in model.named_modules()
        if _is_quantized_linear(mod)
    ]

    if not target_modules:
        logger.warning("No quantized Linears found — calibration is a no-op.")
        return activation_max

    logger.info("Calibrating input_global_scale on %d quantized Linears", len(target_modules))

    def _make_pre_hook(mod_name: str) -> Callable:
        def _pre_hook(_mod: Any, args: tuple[Any, ...]) -> None:
            if not args:
                return
            x = args[0]
            if x is None or not isinstance(x, torch.Tensor):
                return
            # Skip empty inputs (e.g. MoE experts that received zero routed
            # tokens this batch — x has shape (0, hidden_dim) and .max() fails).
            if x.numel() == 0:
                return
            with torch.no_grad():
                cur_max = float(x.detach().float().abs().max().item())
            prev = activation_max.get(mod_name, 0.0)
            if cur_max > prev:
                activation_max[mod_name] = cur_max
        return _pre_hook

    hooks = []
    for name, mod in target_modules:
        hooks.append(mod.register_forward_pre_hook(_make_pre_hook(name)))

    try:
        model.eval()

        if isinstance(calibration_tensor, (str, os.PathLike)):
            calibration_tensor = torch.load(calibration_tensor)

        n_total = calibration_tensor.shape[0]
        n = n_total if num_samples is None else min(num_samples, n_total)
        logger.info("Forward pass over %d/%d calibration samples (batch=%d)",
                    n, n_total, batch_size)

        device = next(model.parameters()).device

        import time
        t0 = time.time()
        observed_count = 0
        with torch.no_grad():
            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)
                batch = calibration_tensor[i:end].to(device)
                try:
                    _ = model(input_ids=batch, use_cache=False)
                except torch.cuda.OutOfMemoryError as oom:
                    logger.error("OOM at sample %d (batch_size=%d). Reduce batch_size or "
                                 "tighten max_memory cap. Error: %s", i, batch_size, oom)
                    raise
                except Exception as e:
                    # Non-OOM forward errors: log and skip the sample rather than abort
                    logger.warning("Calibration forward failed at sample %d: %s", i, e)
                    continue
                observed_count = sum(1 for v in activation_max.values() if v > 0)
                done = end
                if (done // batch_size) % log_every == 0 or done == n:
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 0.001)
                    eta = (n - done) / max(rate, 0.001)
                    logger.info("  %d/%d samples done (%.2fs/sample, ETA %.0fs, %d/%d hooks live)",
                                done, n, elapsed / max(done, 1), eta,
                                observed_count, len(target_modules))
        logger.info("Calibration forward complete in %.0fs", time.time() - t0)
    finally:
        for h in hooks:
            h.remove()

    # Set input_global_scale per module
    missing = []
    set_count = 0
    for name, mod in target_modules:
        max_act = activation_max.get(name, 0.0)
        if max_act <= 0:
            missing.append(name)
            continue
        scale = max_act / FP4_E2M1_MAX
        # Store as fp32 buffer; the loader in CutlassNvFp4LinearKernel reads
        # this as input_global_scale_inv = 1 / input_global_scale during
        # process_weights_after_loading.
        import torch
        mod.input_global_scale = torch.nn.Parameter(
            torch.tensor(scale, dtype=torch.float32), requires_grad=False
        )
        set_count += 1

    logger.info("Set input_global_scale on %d/%d Linears (%d had zero activations)",
                set_count, len(target_modules), len(missing))
    if missing:
        logger.warning("Missing observations on %d modules — first few: %s",
                       len(missing), missing[:5])

    return activation_max


def calibrate_input_global_scales_layerwise(
    model: Any,
    calibration_tensor: Any,
    batch_size: int = 1,
    num_samples: int | None = None,
    device: str = "cuda:0",
    max_layer: int | None = None,
    log_every_layer: int = 5,
) -> dict[str, float]:
    """Layer-wise GPU calibration for Zaya. Works around two issues with the
    naive ``calibrate_input_global_scales``:

    - Zaya's CCA attention calls CUDA-only primitives → full-model CPU forward
      crashes (silently caught by the prior implementation, leaving 99% of
      Linears with zero observations).
    - 8B BF16 model (~16 GB) doesn't fit alongside activations in 16 GB VRAM
      → full-model GPU load isn't viable either.

    Algorithm:
      1. Move ``embed_tokens`` (+ rotary, + final_norm) to GPU. Embed all
         calibration samples on GPU, cache initial hidden_states to CPU.
      2. For each decoder layer 0..max_layer:
         a. Move layer to GPU
         b. Register forward-pre-hooks on every quantized Linear in this layer
         c. For each cached sample state (hidden, residual, router), move to
            GPU, run layer forward, cache new state to CPU
         d. Remove hooks, move layer back to CPU, free VRAM
      3. Compute ``input_global_scale = max_act / FP4_E2M1_MAX`` per Linear
         and attach as a ``torch.nn.Parameter``.

    Model must already be on CPU with ``quantization_scheme`` attached to each
    target Linear (i.e. ``apply_quantization_config`` has been run).

    Returns the raw max-activation dict (keyed by full module name) for
    diagnostics. Hooks are removed before return.
    """
    import time

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Layer-wise GPU calibration requires CUDA")

    dev = torch.device(device)

    activation_max: dict[str, float] = {}

    def _make_pre_hook(mod_name: str) -> Callable:
        def _pre_hook(_mod: Any, args: tuple[Any, ...]) -> None:
            if not args:
                return
            x = args[0]
            if x is None or not isinstance(x, torch.Tensor):
                return
            # Skip empty inputs (MoE experts that received zero routed tokens
            # this batch — x has shape (0, hidden_dim) and .max() fails).
            if x.numel() == 0:
                return
            with torch.no_grad():
                cur_max = float(x.detach().float().abs().max().item())
            prev = activation_max.get(mod_name, 0.0)
            if cur_max > prev:
                activation_max[mod_name] = cur_max
        return _pre_hook

    if isinstance(calibration_tensor, (str, os.PathLike)):
        calibration_tensor = torch.load(calibration_tensor)

    n_total = calibration_tensor.shape[0]
    n = n_total if num_samples is None else min(num_samples, n_total)
    seq_len = calibration_tensor.shape[1]
    logger.info("Layer-wise calibration: %d/%d samples × %d tokens, batch=%d",
                n, n_total, seq_len, batch_size)

    # ── Walk the model: embed, layers list, rotary modules ─────
    base = model.model if hasattr(model, "model") else model
    embed = base.embed_tokens
    layers = base.layers
    rotary = getattr(base, "rotary_emb", None)
    swa_rotary = getattr(base, "swa_rotary_emb", None)
    swa_layers = getattr(model.config, "swa_layers", None)

    num_layers = len(layers)
    if max_layer is None:
        max_layer = num_layers
    max_layer = min(max_layer, num_layers)
    logger.info("Processing layers 0..%d (of %d total)", max_layer - 1, num_layers)

    # ── Pre-compute shared per-sample state on GPU ─────────────
    # All calibration samples are [1, seq_len] int with no padding, so
    # position_ids, causal_mask, and rotary outputs are identical across
    # samples. Compute once.
    embed.to(dev)
    if rotary is not None:
        rotary.to(dev)
    if swa_rotary is not None:
        swa_rotary.to(dev)

    with torch.no_grad():
        # Embed all samples in batches → list of CPU bf16 [batch, seq, hidden]
        hidden_states_cpu: list[torch.Tensor] = []
        t0 = time.time()
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            batch_ids = calibration_tensor[i:end].to(dev)
            h = embed(batch_ids)
            hidden_states_cpu.append(h.detach().cpu())
        logger.info("Embedded %d samples in %.1fs", n, time.time() - t0)

        # Shared position state. All calibration samples are [1, seq_len] with
        # no padding, so position_ids / cache_position / rotary outputs are
        # identical across samples. cca_mask=None when there's no padding.
        dummy_h = hidden_states_cpu[0].to(dev)
        cache_position = torch.arange(seq_len, device=dev)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = rotary(dummy_h, position_ids) if rotary is not None else None
        swa_position_embeddings = (
            swa_rotary(dummy_h, position_ids) if swa_rotary is not None else None
        )
        del dummy_h

    embed.to("cpu")
    if rotary is not None:
        rotary.to("cpu")
    if swa_rotary is not None:
        swa_rotary.to("cpu")
    torch.cuda.empty_cache()

    # Per-sample state: (hidden, residual, prev_router_hidden_states) — all CPU.
    # `residual` and `prev_router_hidden_states` start as None for every sample.
    residual_cpu: list[Any] = [None] * len(hidden_states_cpu)
    router_cpu: list[Any] = [None] * len(hidden_states_cpu)

    # ── Iterate layers ─────────────────────────────────────────
    t_start = time.time()
    for layer_idx in range(max_layer):
        layer = layers[layer_idx]
        layer.to(dev)

        # Pick rotary embeddings (sliding-window vs full)
        if swa_position_embeddings is not None and swa_layers is not None:
            emb_to_use = (position_embeddings if swa_layers[layer_idx] == 0
                          else swa_position_embeddings)
        else:
            emb_to_use = position_embeddings

        # Register hooks on quantized Linears within this layer
        local_hooks = []
        for name, mod in layer.named_modules():
            if _is_quantized_linear(mod):
                full_name = f"model.layers.{layer_idx}.{name}" if name else f"model.layers.{layer_idx}"
                local_hooks.append(mod.register_forward_pre_hook(_make_pre_hook(full_name)))

        # Forward each sample through this layer
        with torch.no_grad():
            for sample_idx in range(len(hidden_states_cpu)):
                h = hidden_states_cpu[sample_idx].to(dev)
                r = residual_cpu[sample_idx]
                if r is not None:
                    r = r.to(dev)
                pr = router_cpu[sample_idx]
                if pr is not None:
                    pr = pr.to(dev)

                try:
                    layer_outputs, r_new, pr_new = layer(
                        h,
                        r,
                        attention_mask=None,
                        position_ids=position_ids,
                        past_key_values=None,
                        output_attentions=False,
                        use_cache=False,
                        cache_position=cache_position,
                        position_embeddings=emb_to_use,
                        prev_router_hidden_states=pr,
                        cca_mask=None,
                    )
                except torch.cuda.OutOfMemoryError as oom:
                    logger.error("OOM at layer %d sample %d: %s", layer_idx, sample_idx, oom)
                    raise

                h_new = layer_outputs[0]
                hidden_states_cpu[sample_idx] = h_new.detach().cpu()
                residual_cpu[sample_idx] = r_new.detach().cpu() if r_new is not None else None
                router_cpu[sample_idx] = pr_new.detach().cpu() if pr_new is not None else None

                del h, r, pr, h_new, r_new, pr_new, layer_outputs

        for hk in local_hooks:
            hk.remove()
        layer.to("cpu")
        torch.cuda.empty_cache()

        if (layer_idx + 1) % log_every_layer == 0 or layer_idx == max_layer - 1:
            elapsed = time.time() - t_start
            done = layer_idx + 1
            eta = elapsed * (max_layer - done) / max(done, 1)
            observed = sum(1 for v in activation_max.values() if v > 0)
            logger.info("  layer %d/%d done (%.1fs elapsed, ETA %.0fs, %d hooks fired)",
                        done, max_layer, elapsed, eta, observed)

    logger.info("Layer-wise forward complete in %.0fs", time.time() - t_start)

    # ── Set input_global_scale per quantized Linear ────────────
    # Only set on modules WITHIN the processed layer range so dry-run output
    # matches dry-run expectations.
    target_modules: list[tuple[str, Any]] = []
    for name, mod in model.named_modules():
        if not _is_quantized_linear(mod):
            continue
        layer_idx = _extract_layer_idx(name)
        if layer_idx is not None and layer_idx >= max_layer:
            continue
        target_modules.append((name, mod))

    import torch as _torch
    missing = []
    set_count = 0
    for name, mod in target_modules:
        max_act = activation_max.get(name, 0.0)
        if max_act <= 0:
            missing.append(name)
            continue
        scale = max_act / FP4_E2M1_MAX
        mod.input_global_scale = _torch.nn.Parameter(
            _torch.tensor(scale, dtype=_torch.float32), requires_grad=False
        )
        set_count += 1

    logger.info("Set input_global_scale on %d/%d Linears (%d had zero activations)",
                set_count, len(target_modules), len(missing))
    if missing:
        logger.warning("Missing observations on %d modules — first few: %s",
                       len(missing), missing[:5])

    return activation_max


def _extract_layer_idx(module_name: str) -> int | None:
    """Pull the integer N from a name like 'model.layers.N.foo.bar'."""
    parts = module_name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def run_w4a4(args: Any) -> int:
    """Full NVFP4 W4A4 quantization pipeline (Path B Phase 1).

    Differs from the W4A16 pipeline:
      1. Loads model with device_map="auto" + memory cap (need GPU for calibration forward)
      2. Uses preset_name_to_scheme("NVFP4") and adds W4A4_IGNORE_PATTERNS to config
      3. Inserts activation calibration step before weight scale compute
      4. Saves input_global_scale per Linear in the output state dict
    """
    import time
    from pathlib import Path

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
    logger.info("STAGE 1: NVFP4 W4A4 Quantization (Path B Phase 1)")
    logger.info("Model: %s | Output: %s | Mode: %s",
                args.model_id, args.output_dir,
                f"DRY RUN ({DRY_RUN_LAYERS_W4A4:d}L)" if args.dry_run else "FULL")
    logger.info("Calibration: %s | samples: %s | batch: %d",
                args.calibration_data,
                args.calibration_num_samples or "all", args.calibration_batch_size)
    logger.info("Ignore patterns: %s", W4A4_IGNORE_PATTERNS)
    logger.info("torch %s | CUDA: %s | %s",
                torch.__version__, torch.cuda.is_available(),
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
    logger.info("compressed_tensors %s", getattr(ct, "__version__", "?"))
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA required for W4A4 calibration forward pass")
        return 1

    cal_path = Path(args.calibration_data)
    if not cal_path.exists():
        logger.error("Calibration data not found: %s. Run scripts/build_calibration_data.py first.",
                     cal_path)
        return 1

    # ── Load model on CPU ───────────────────────────────────
    # NOTE: We can't use device_map="auto" here. compressed_tensors'
    # set_forward_quantized assumes module.forward is a bound method, but
    # accelerate's device_map dispatch wraps forward in a functools.partial,
    # which causes apply_quantization_config to fail with
    # "AttributeError: 'functools.partial' object has no attribute '__func__'".
    # Loading on CPU keeps module.forward as a plain bound method.
    #
    # Calibration forward runs on CPU as a consequence (~30s per 1024-token
    # sample for 8B BF16). For the full 979-sample run that's ~8 GPU-hr-
    # equivalent. Acceptable for once-off Phase 1. Layer-wise GPU calibration
    # is a future optimization (load+forward+free one layer at a time).
    logger.info("Loading model on CPU (no device_map — avoids compressed_tensors conflict)...")
    t0 = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    logger.info("Model loaded in %.0fs | Params: %.1fB", time.time() - t0,
                sum(p.numel() for p in model.parameters()) / 1e9)

    # Generation config sanity (matches W4A16 path)
    gc = model.generation_config
    if not gc.do_sample and gc.top_p is not None:
        gc.top_p = None
    if not gc.do_sample and gc.top_k is not None:
        gc.top_k = None

    # ── Apply NVFP4 (W4A4) scheme with ignore list ──────────
    scheme = preset_name_to_scheme("NVFP4", targets=["Linear"])
    logger.info("Scheme: W=%s/g=%s, A=%s/g=%s (dyn=%s)",
                scheme.weights.num_bits, scheme.weights.group_size,
                scheme.input_activations.num_bits, scheme.input_activations.group_size,
                scheme.input_activations.dynamic)

    config = QuantizationConfig(
        quant_method="compressed-tensors",
        format="float-quantized",
        config_groups={"group_0": scheme},
        ignore=W4A4_IGNORE_PATTERNS,
    )
    apply_quantization_config(model, config)
    logger.info("Quantization config applied (NVFP4 W4A4 scheme + ignore list)")

    # apply_quantization_config replaces each quantized Linear's forward with a
    # fake-quant wrapper that uses weight_scale (initialized to NaN) and
    # input_global_scale (initialized to ~0). With those defaults the wrapper
    # outputs NaN, which silently corrupts downstream activations and means
    # only the FIRST layer of pre-hooks observes real values — every hook past
    # the first quantized Linear sees zeros/NaN.
    #
    # Restore plain nn.Linear.forward on each quantized Linear so calibration
    # runs in pure BF16. We compute scales after this; the fake-quant wrapper
    # is no longer needed because the model is never forwarded post-calibration
    # (we extract weights for the compressor and save).
    import types

    import torch.nn as _nn
    restored = 0
    for _name, _mod in model.named_modules():
        if isinstance(_mod, _nn.Linear) and hasattr(_mod, "quantization_scheme"):
            _mod.forward = types.MethodType(_nn.Linear.forward, _mod)
            restored += 1
    logger.info("Restored plain Linear.forward on %d quantized Linears for BF16 calibration",
                restored)

    # ── Activation calibration ──────────────────────────────
    logger.info("Loading calibration tensor: %s", cal_path)
    cal_tensor = torch.load(cal_path)
    logger.info("Calibration shape: %s, dtype: %s", list(cal_tensor.shape), cal_tensor.dtype)

    if args.dry_run:
        # Keep the calibration cheap during dry-run
        cal_tensor = cal_tensor[: min(8, cal_tensor.shape[0])]
        logger.info("DRY RUN: truncated calibration to %d samples", cal_tensor.shape[0])

    # Layer-wise GPU calibration — full-model CPU forward crashes for Zaya
    # (CCA attention requires CUDA). 8B BF16 doesn't fit in 16 GB VRAM as a
    # whole, so we move one decoder layer at a time on/off GPU.
    max_layer = DRY_RUN_LAYERS_W4A4 if args.dry_run else None
    activation_max = calibrate_input_global_scales_layerwise(
        model=model,
        calibration_tensor=cal_tensor,
        batch_size=args.calibration_batch_size,
        num_samples=args.calibration_num_samples,
        max_layer=max_layer,
    )

    # Diagnostic: distribution of input_global_scales
    if activation_max:
        import statistics as stats
        vals = sorted(activation_max.values())
        logger.info("Activation max distribution across Linears:")
        logger.info("  min: %.4f | p25: %.4f | median: %.4f | p75: %.4f | max: %.4f",
                    vals[0], vals[len(vals) // 4], stats.median(vals),
                    vals[3 * len(vals) // 4], vals[-1])

    # ── Compute weight scales (same logic as W4A16) ─────────
    logger.info("Computing per-group weight scales...")
    calibrated_w = 0
    skipped_layers = []
    num_layers = model.config.num_hidden_layers
    max_idx = min(num_layers, DRY_RUN_LAYERS_W4A4) if args.dry_run else num_layers

    for name, module in model.named_modules():
        qscheme = getattr(module, "quantization_scheme", None)
        if qscheme is None or qscheme.weights is None:
            continue

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
                skipped_layers.append(name)
                continue

        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        gs = qscheme.weights.group_size
        w = weight.data.float()
        out, in_feat = w.shape
        w_groups = w.view(out, in_feat // gs, gs)
        w_max = w_groups.abs().amax(dim=-1)
        scales = w_max / FP4_E2M1_MAX
        scales = torch.clamp(scales, min=1e-12)
        module.weight_scale.data = scales.to(torch.bfloat16)
        calibrated_w += 1

    logger.info("Calibrated %d weight scales (%d layers skipped by dry-run)",
                calibrated_w, len(skipped_layers))

    # ── Compress each quantized module ──────────────────────
    compressed_params: dict[str, dict[str, torch.Tensor]] = {}
    linear_count = 0
    skipped = 0

    logger.info("Compressing weights with NVFP4PackedCompressor...")
    t0 = time.time()

    for name, module in model.named_modules():
        qscheme = getattr(module, "quantization_scheme", None)
        if qscheme is None or qscheme.weights is None:
            continue
        if args.dry_run and name in skipped_layers:
            continue

        weight = getattr(module, "weight", None)
        scale = getattr(module, "weight_scale", None)
        if weight is None or scale is None:
            skipped += 1
            continue

        linear_count += 1
        state_dict: dict[str, torch.Tensor] = {
            "weight": weight.data.to("cuda:0"),
            "weight_scale": scale.data.to("cuda:0"),
        }
        compressed = NVFP4PackedCompressor.compress(state_dict, qscheme)
        compressed_params[name] = {k: v for k, v in compressed.items()}  # noqa: C416

        # Attach the per-Linear input_global_scale (fp32) — saved alongside the
        # packed weight in the output state dict below.
        igs = getattr(module, "input_global_scale", None)
        if igs is not None:
            compressed_params[name]["input_global_scale"] = igs.data.to(torch.float32).cpu()

    logger.info("Compressed %d Linear modules in %.0fs (skipped %d)",
                linear_count, time.time() - t0, skipped)

    # ── Build output state dict ─────────────────────────────
    # Same shape as W4A16 path, but with input_global_scale per Linear.
    # Dry-run bloat fix: parameters belonging to layers beyond the dry-run
    # cutoff (norms, routers, full expert MLPs, etc.) get dropped instead of
    # saved as raw BF16. Saving them duplicates ~16 GB of unquantized weights
    # and gives a misleading 18 GB dry-run output.
    output_state: dict[str, torch.Tensor] = {}
    packed_total = 0
    scale_total = 0
    igs_total = 0
    other_total = 0
    dropped_dryrun = 0
    dryrun_cutoff = DRY_RUN_LAYERS_W4A4 if args.dry_run else None

    for pname, param in model.named_parameters():
        module_name = ".".join(pname.split(".")[:-1])
        param_short = pname.split(".")[-1]

        if dryrun_cutoff is not None:
            li = _extract_layer_idx(pname)
            if li is not None and li >= dryrun_cutoff:
                dropped_dryrun += 1
                continue

        if module_name in compressed_params:
            if param_short == "weight":
                continue
            if param_short == "weight_zero_point":
                continue
            if param_short in compressed_params[module_name]:
                tensor = compressed_params[module_name][param_short]
                output_state[pname] = tensor.cpu() if tensor.device.type != "cpu" else tensor
                if "weight_packed" in param_short:
                    packed_total += tensor.numel() * tensor.element_size()
                elif "input_global_scale" in param_short:
                    igs_total += tensor.numel() * tensor.element_size()
                elif "weight_scale" in param_short and "global" not in param_short:
                    scale_total += tensor.numel() * tensor.element_size()
                continue
            if param.device.type != "meta":
                output_state[pname] = param.detach().cpu()
                other_total += param.numel() * param.element_size()
            continue

        if param.device.type != "meta":
            output_state[pname] = param.detach().cpu()
            other_total += param.numel() * param.element_size()

    if dropped_dryrun:
        logger.info("DRY RUN: dropped %d params from layers ≥%d (avoids ~16 GB BF16 bloat)",
                    dropped_dryrun, dryrun_cutoff)

    # Add every compressed-param tensor (including input_global_scale) that
    # didn't already get pulled in by the named_parameters loop.
    for mod_name, comp in compressed_params.items():
        for key, tensor in comp.items():
            full_name = f"{mod_name}.{key}"
            if full_name not in output_state:
                output_state[full_name] = tensor.cpu() if tensor.device.type != "cpu" else tensor
                if "weight_packed" in key:
                    packed_total += tensor.numel() * tensor.element_size()
                elif "input_global_scale" in key:
                    igs_total += tensor.numel() * tensor.element_size()

    logger.info("Output: %d params | packed: %.0f MB | weight_scales: %.0f MB | "
                "input_global_scale: %.1f KB | other: %.0f MB",
                len(output_state), packed_total / 1e6, scale_total / 1e6,
                igs_total / 1e3, other_total / 1e6)

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
            "scheme": "NVFP4",
            "scheme_variant": "w4a4",
            "num_bits_weight": 4,
            "num_bits_activation": 4,
            "type": "float",
            "strategy": "tensor_group",
            "group_size": 16,
            "target_modules": ["Linear"],
            "ignore": W4A4_IGNORE_PATTERNS,
            "weight_packed_dtype": "uint8",
            "weight_scale_dtype": "float8_e4m3fn",
            "weight_global_scale_dtype": "float32",
            "input_global_scale_dtype": "float32",
            "zero_point": "none (symmetric)",
            "compressor": "NVFP4PackedCompressor",
        },
        "calibration": {
            "data_path": str(cal_path),
            "num_samples_used": args.calibration_num_samples or cal_tensor.shape[0],
            "batch_size": args.calibration_batch_size,
            "linears_calibrated": sum(1 for n, _ in [(k, v) for k, v in compressed_params.items()
                                                     if "input_global_scale" in v]),
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

    logger.info("Saved in %.0fs | Total: %.0f MB (%.2f GB)",
                time.time() - t0, total_bytes / 1e6, total_bytes / 1e9)

    # ── Verify ──────────────────────────────────────────────
    logger.info("")
    logger.info("=== Verification ===")
    verify = st.load_file(str(output_dir / "model.safetensors"), device="cpu")
    igs_keys = [k for k in verify if "input_global_scale" in k]
    logger.info("input_global_scale keys present: %d", len(igs_keys))

    # Quality gate: input_global_scale coverage must be ≥95% of compressed Linears.
    # The dry-run failure mode that this catches: forward pass crashes silently,
    # ~4 of ~1480 modules get scales (coverage <0.3%), verification "passes"
    # because IGS keys exist with garbage values (8.97e-44).
    expected_igs = linear_count
    coverage = (len(igs_keys) / expected_igs) if expected_igs > 0 else 0.0
    nonzero_igs = 0
    near_zero_igs = 0
    sample_vals: list[float] = []
    for k in igs_keys:
        v = verify[k]
        if v.numel() != 1:
            continue
        val = float(v.item())
        if abs(val) > 1e-12:
            nonzero_igs += 1
            if len(sample_vals) < 5:
                sample_vals.append(val)
        if abs(val) < 1e-30:
            near_zero_igs += 1
    nonzero_coverage = (nonzero_igs / expected_igs) if expected_igs > 0 else 0.0

    logger.info("IGS coverage: %d/%d keys present (%.1f%%), %d nonzero (%.1f%%), "
                "%d near-zero garbage", len(igs_keys), expected_igs, coverage * 100,
                nonzero_igs, nonzero_coverage * 100, near_zero_igs)
    if sample_vals:
        logger.info("Sample IGS values: %s", [f"{v:.4f}" for v in sample_vals])

    # Threshold: ≥95% of compressed Linears must have a non-zero input_global_scale.
    # Anything lower means calibration didn't reach those modules.
    coverage_threshold = 0.95
    if nonzero_coverage < coverage_threshold:
        logger.error("QUALITY GATE FAILED: input_global_scale coverage %.1f%% < %.0f%% threshold.",
                     nonzero_coverage * 100, coverage_threshold * 100)
        logger.error("This indicates calibration forward pass did not exercise most quantized "
                     "Linears. Common causes: model on wrong device, layer-wise iteration "
                     "stopped early, or hooks failed to register.")
        return 2

    logger.info("")
    logger.info("=" * 60)
    if args.dry_run:
        logger.info("W4A4 DRY RUN PASSED — pipeline verified on %d layers", DRY_RUN_LAYERS_W4A4)
        logger.info("Ready for full: python scripts/quantize_zaya_ct_nvfp4.py --scheme w4a4")
    else:
        logger.info("W4A4 COMPLETE — NVFP4 W4A4 ZAYA1-8B")
        logger.info("Output: %s (%.1f GB)", output_dir, total_bytes / 1e9)
        logger.info("Next: smoke test with VLLM_NVFP4_GEMM_BACKEND=cutlass to force SM120 CUTLASS")
    logger.info("=" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1: ZAYA1-8B → NVFP4 compressed-tensors")
    parser.add_argument("--scheme", choices=["w4a16", "w4a4"], default="w4a16",
                        help="Quantization scheme: w4a16 (legacy SOTA) or w4a4 (Path B Phase 1)")
    parser.add_argument("--model-id", default=os.environ.get("ZAYA_MODEL_ID", DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=None,
                        help=f"Output directory (default: {DEFAULT_OUTPUT} for w4a16, "
                             f"{DEFAULT_OUTPUT_W4A4} for w4a4)")
    parser.add_argument("--dry-run", action="store_true",
                        help=f"Quantize only first {DRY_RUN_LAYERS} (w4a16) / "
                             f"{DRY_RUN_LAYERS_W4A4} (w4a4) layers")
    parser.add_argument("--calibration-data", default=DEFAULT_CALIBRATION_DATA,
                        help="Path to calibration_data.pt (w4a4 only). "
                             f"Default: {DEFAULT_CALIBRATION_DATA}")
    parser.add_argument("--calibration-num-samples", type=int, default=None,
                        help="Cap calibration to first N samples for quick iteration. "
                             "Default: use the entire calibration tensor.")
    parser.add_argument("--calibration-batch-size", type=int, default=1,
                        help="Batch size for calibration forward. Default 1 — 8B BF16 + "
                             "1024-token activations near saturate 16 GB.")
    args = parser.parse_args()

    # Pick a sensible default output_dir per scheme if user didn't override
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_W4A4 if args.scheme == "w4a4" else DEFAULT_OUTPUT

    if args.scheme == "w4a4":
        return run_w4a4(args)

    # ── W4A16 path below (unchanged from original) ──────────

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
