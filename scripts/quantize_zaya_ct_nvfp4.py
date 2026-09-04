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
    hooks, then sets input_global_scale = 2688 / max_act.
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
import gc
import json
import logging
import os
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Zyphra/ZAYA1-8B-legacy"
DEFAULT_OUTPUT = "./zaya1-8b-nvfp4-ct"
DEFAULT_OUTPUT_W4A4 = "./zaya1-8b-nvfp4-w4a4"
DEFAULT_CALIBRATION_DATA = "data/calibration/calibration_data.pt"
DRY_RUN_LAYERS = 2
DRY_RUN_LAYERS_W4A4 = 4  # need a few more for hook coverage to make sense
FP4_E2M1_MAX = 6.0  # max magnitude representable in FP4 E2M1
FP8_E4M3_MAX = 448.0  # max magnitude representable in FP8 E4M3
# CT convention: global_scale = FP8_MAX * FP4_MAX / max_abs = 2688 / max_abs
# (see compressed_tensors.quantization.utils.helpers._compute_global_scale)
GLOBAL_SCALE_NUM = FP8_E4M3_MAX * FP4_E2M1_MAX  # 2688.0

# W4A4 layers that MUST stay BF16 (regex patterns for compressed_tensors
# ignore list). compressed_tensors matches against module *paths* from
# named_modules(), not class names — so the CCA pattern must target the
# `qkv` attribute that ZayaAttention uses for its CCA submodule (paths look
# like `model.layers.0.self_attn.qkv.linear_q`). The earlier `re:.*cca.*`
# matched nothing and silently quantized 160 CCA projections to W4A4.
W4A4_IGNORE_PATTERNS = [
    "lm_head",  # tied to embed_tokens, BF16
    "re:.*router.*",  # MoE router (size_n=17, doesn't fit FP4 grid cleanly)
    "re:.*norm.*",  # RMSNorm
    "re:.*qkv.*",  # named_modules path: model.layers.X.self_attn.qkv.linear_q
    "re:.*cca.*",  # construct-time prefix path: model.layers.X.self_attn.cca.linear_q
]

# Default threshold for dynamic mixed-precision exemption. MoE layers where any
# expert activation max_abs exceeds this value are kept at BF16 rather than W4A4.
# Derived from outlier analysis: 24 linear_fc2 modules have max_abs > 500, with
# the top offender at 8896 (L75.experts.1) causing 3.31× effective FP8 saturation.
# FusedMoE requires uniform quantization across all experts in a layer, so we
# exempt the entire layer's MLP when any single expert is an outlier.
MIXED_PRECISION_DEFAULT_THRESHOLD = 500.0


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
        (name, mod) for name, mod in model.named_modules() if _is_quantized_linear(mod)
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
        logger.info("Forward pass over %d/%d calibration samples (batch=%d)", n, n_total, batch_size)

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
                    logger.error(
                        "OOM at sample %d (batch_size=%d). Reduce batch_size or tighten max_memory cap. Error: %s",
                        i,
                        batch_size,
                        oom,
                    )
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
                    logger.info(
                        "  %d/%d samples done (%.2fs/sample, ETA %.0fs, %d/%d hooks live)",
                        done,
                        n,
                        elapsed / max(done, 1),
                        eta,
                        observed_count,
                        len(target_modules),
                    )
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
        # CT convention: input_global_scale = 2688 / max_act (the DIVISOR form).
        # The loader passes this directly to the kernel as the divisor,
        # computing per-block scale = input_global_scale * block_max / 6.0.
        scale = GLOBAL_SCALE_NUM / max_act
        import torch

        mod.input_global_scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float32), requires_grad=False)
        set_count += 1

    logger.info(
        "Set input_global_scale on %d/%d Linears (%d had zero activations)",
        set_count,
        len(target_modules),
        len(missing),
    )
    if missing:
        logger.warning("Missing observations on %d modules — first few: %s", len(missing), missing[:5])

    return activation_max


# ── MSE-optimized calibration helpers ──────────────────────────────
# NVFP4 representable values (symmetric, positive half)
_NVFP4_LEVELS = torch.tensor(
    [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def _fake_quant_nvfp4(x: torch.Tensor, input_global_scale: float) -> torch.Tensor:
    """Fake-quantize activations to NVFP4 with a given input_global_scale.

    Used offline in ``_optimize_input_global_scales`` to evaluate candidate
    scales without running the model forward again.
    """
    orig_shape = x.shape
    gs = 16
    x_2d = x.reshape(-1, orig_shape[-1])  # [N, D]
    # Reshape to groups of 16
    x_g = x_2d.reshape(-1, gs)  # [N * n_g, 16]
    # Per-group max
    g_max = torch.clamp(x_g.abs().amax(dim=-1, keepdim=True), min=1e-12)
    # Block scale (same per-group): [N * n_g, 1]
    bs = torch.clamp(
        input_global_scale * g_max / FP4_E2M1_MAX,
        max=FP8_E4M3_MAX,
    )
    # Normalize
    x_n = x_g / bs
    x_c = torch.clamp(x_n, -FP4_E2M1_MAX, FP4_E2M1_MAX).float()
    # Round to nearest NVFP4 level (non-uniform)
    fp4 = _NVFP4_LEVELS.to(x.device)
    x_abs = x_c.abs()
    dist = (x_abs.unsqueeze(-1) - fp4.unsqueeze(0)).abs()
    best = dist.argmin(dim=-1)
    x_q = fp4[best] * x_c.sign()
    return (x_q * bs).reshape(orig_shape)


def _optimize_input_global_scales(
    activation_cache: dict[str, torch.Tensor],
    activation_max: dict[str, float],
) -> dict[str, float]:
    """Search optimal input_global_scale per Linear by minimizing MSE.

    Generates candidate scales from percentiles and multipliers, picks the one
    with lowest fake-quant MSE on the cached calibration activations.
    """
    optimized: dict[str, float] = {}

    for name, max_act in activation_max.items():
        if max_act <= 0:
            continue
        cached = activation_cache.get(name)
        if cached is None or cached.numel() == 0:
            optimized[name] = GLOBAL_SCALE_NUM / max_act
            continue

        cached = cached.float()
        base_scale = GLOBAL_SCALE_NUM / max_act

        # Build candidate list
        candidates: list[float] = [base_scale]
        vals_flat = cached.abs().flatten()
        if vals_flat.numel() > 0:
            for p in [90, 95, 99]:
                pv = float(torch.quantile(vals_flat, p / 100.0).item())
                if pv > 0:
                    candidates.append(GLOBAL_SCALE_NUM / pv)
        for mult in [0.5, 0.75, 1.25, 1.5]:
            candidates.append(base_scale * mult)

        best_scale = base_scale
        best_mse = float("inf")
        for c in set(candidates):  # deduplicate
            dq = _fake_quant_nvfp4(cached, c)
            mse = float(torch.mean((cached - dq) ** 2).item())
            if mse < best_mse:
                best_mse = mse
                best_scale = c
        optimized[name] = best_scale

    return optimized


def _compute_soar_global_scale(
    max_act: float,
    block_maxes: list[torch.Tensor],
    group_size: int = 16,
    n_candidates: int = 25,
) -> float:
    """SOAR closed-form scale optimization (arXiv:2605.12245).

    Finds the global_scale minimizing FP8 block-scale rounding error across
    the calibration data, rather than just fitting the max activation.

    The key insight: at global_scale g, each FP4 block's FP8 scale is
    ``fp8_raw = block_max / (g * FP4_MAX)`` rounded to the nearest FP8 value.
    The rounding error ``(fp8_rounded - fp8_raw)^2 * block_max^2`` is the
    dominant quantization error term beyond max-abs coverage. Minimizing this
    over a log-spaced grid of candidates gives better accuracy than max-abs
    without ever clipping any activation values.

    Args:
        max_act:     Overall max activation magnitude (from max-abs calibration).
        block_maxes: List of 1-D tensors, each holding per-block max values
                     computed from reservoir-sampled activation vectors.
        group_size:  NVFP4 block size (default 16).
        n_candidates: Grid resolution (default 25).

    Returns:
        Optimal global_scale (float32).
    """
    import math

    base_scale = GLOBAL_SCALE_NUM / max_act
    if not block_maxes:
        return base_scale

    # Concatenate all block-max samples: shape [N_blocks_total]
    bm = torch.cat(block_maxes).float()
    if bm.numel() == 0:
        return base_scale

    # Log-spaced candidates: 0.5× to 2.0× the max-abs baseline
    lo = math.log10(base_scale * 0.5)
    hi = math.log10(base_scale * 2.0)
    candidates = torch.logspace(lo, hi, n_candidates, dtype=torch.float32)

    best_scale = base_scale
    best_err = float("inf")

    for g in candidates.tolist():
        # FP8 block scales at this global scale
        fp8_raw = (bm / (g * FP4_E2M1_MAX)).clamp(1e-38, FP8_E4M3_MAX)
        # Round to nearest FP8 E4M3
        fp8_rounded = fp8_raw.to(torch.float8_e4m3fn).to(torch.float32)
        # Weighted relative error: (Δs / s)² × block_max²
        rel_err = ((fp8_rounded - fp8_raw) / fp8_raw.clamp(min=1e-38)) ** 2
        weighted_err = float((rel_err * bm**2).sum().item())
        if weighted_err < best_err:
            best_err = weighted_err
            best_scale = g

    return best_scale


# ─────────────────────────────────────────────────────────────────────────────
# MR-GPTQ helpers (arXiv:2509.23202 — Micro-Rotation GPTQ)
# ─────────────────────────────────────────────────────────────────────────────


def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round each element to the nearest FP4 E2M1 representable value.

    Input x is assumed to be prescaled so that the target range is [-6, 6].
    FP4 E2M1 positive values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
    Uses round-to-nearest-even at the midpoints between adjacent values.
    """
    sign = x.sign()
    ax = x.abs()
    q = torch.zeros_like(ax)
    q = torch.where(ax > 0.25, torch.full_like(ax, 0.5), q)
    q = torch.where(ax > 0.75, torch.full_like(ax, 1.0), q)
    q = torch.where(ax > 1.25, torch.full_like(ax, 1.5), q)
    q = torch.where(ax > 1.75, torch.full_like(ax, 2.0), q)
    q = torch.where(ax > 2.5, torch.full_like(ax, 3.0), q)
    q = torch.where(ax > 3.5, torch.full_like(ax, 4.0), q)
    q = torch.where(ax > 5.0, torch.full_like(ax, 6.0), q)
    return sign * q


def _gptq_correction(
    W: torch.Tensor,
    H_hess: torch.Tensor,
    group_size: int = 16,
    dampening_frac: float = 0.02,
) -> torch.Tensor:
    """Hessian-weighted GPTQ quantization for a single Linear layer.

    Standard GPTQ (Frantar et al., 2022) column-by-column update, adapted for
    FP4 E2M1 with per-row per-group scaling.  For each column j:
      1. Compute the per-row group scale for column j's group.
      2. Quantize W[:, j] to the nearest FP4 value (using that scale).
      3. Propagate the quantization error to remaining columns via H^{-1}.

    Returns W_corrected: float32 tensor of the same shape as W, where each
    value is the nearest FP4-representable value under its per-row group scale.
    The caller should store this back into module.weight.data (as bfloat16);
    NVFP4PackedCompressor.compress() will then pack it losslessly since every
    value is already on the FP4 grid.

    Heavier default dampening (0.02 vs. typical 0.01) guards against near-
    singular Hessians in MoE experts that are only activated by ~60 calibration
    samples.
    """
    n_out, n_in = W.shape
    # Dampen to prevent singular Hessian — important for sparsely-activated
    # MoE experts where H may have near-zero diagonal entries
    damp = dampening_frac * H_hess.diag().mean().clamp(min=1e-6)
    H = H_hess.clone()
    H.diagonal().add_(damp)

    try:
        H_inv = torch.linalg.inv(H)
    except (torch.linalg.LinAlgError, RuntimeError):
        return W  # skip GPTQ if Hessian is singular

    W_q = W.clone()

    for g_start in range(0, n_in, group_size):
        g_end = min(g_start + group_size, n_in)
        # Pre-compute per-row scale for this input-channel group
        # Scale = max_abs(row_group) / FP4_MAX  [out_features, 1]
        w_block = W_q[:, g_start:g_end].float()
        scales = w_block.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / FP4_E2M1_MAX

        for j in range(g_start, g_end):
            w_j = W_q[:, j]  # [out_features]
            # Quantize to nearest FP4 E2M1 value using per-row scale
            w_scaled = (w_j / scales[:, 0]).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)
            w_q_j = _quantize_to_fp4_e2m1(w_scaled) * scales[:, 0]

            quant_err = w_j - w_q_j  # [out_features]

            # GPTQ column update: propagate error to remaining columns
            h_inv_jj = float(H_inv[j, j].item())
            if abs(h_inv_jj) < 1e-12:
                W_q[:, j] = w_q_j
                continue
            if j + 1 < n_in:
                # W[:, j+1:] -= outer(quant_err, H_inv[j, j+1:]) / H_inv[j,j]
                W_q[:, j + 1 :] = W_q[:, j + 1 :] - (
                    quant_err.unsqueeze(1) * (H_inv[j, j + 1 :] / h_inv_jj).unsqueeze(0)
                )
            W_q[:, j] = w_q_j

    return W_q


from compressed_tensors.offload import align_modules
from llmcompressor.modeling.moe.context import moe_calibration_context


class LinearizedExpertsWontMove(RuntimeError):
    """Raised when linearized MoE experts cannot be moved onto the accelerator.

    Measured 2026-09-03 on transformers 5.14.1 / llmcompressor 0.13.0: after
    linearize_moe, the 48 expert weights per layer (16 experts x 3 projections)
    stay on CPU through every documented move. `layer.to(dev)` skips them,
    `param.data = param.data.to(dev)` does not stick, and rebinding
    `setattr(mod, name, nn.Parameter(...))` is silently discarded - while the
    ~28 other parameters in the same layer move normally, and
    `param.data.to("cuda:0")` on its own returns a genuine cuda tensor. So the
    move works; the assignment does not survive on these modules.

    Root cause not established. Rather than guess at a workaround, the caller
    falls back to running calibration entirely on CPU, which needs no moves at
    all. That is the path this script's own docstring describes as the baseline
    ("~30s per 1024-token sample ... acceptable for once-off Phase 1").
    """


def _call_rotary(rotary: Any, hidden: Any, position_ids: Any, layer_type: Any) -> Any:
    """Call a rotary embedding across the transformers 4.x / 5.x signature change.

    transformers >=5 keys the rotary buffers by layer type:
    ``ZayaRotaryEmbedding.forward(x, position_ids, layer_type)`` looks up
    ``f"{layer_type}_inv_freq"``. In 5.14 ``layer_type`` still defaults to None,
    so omitting it does not raise TypeError - it fails later and less obviously
    as::

        AttributeError: 'ZayaRotaryEmbedding' object has no attribute 'None_inv_freq'

    Older transformers take ``(x, position_ids)`` only. Inspect rather than
    pin a version, so this keeps working either way.
    """
    import inspect

    try:
        params = inspect.signature(rotary.forward).parameters
    except (TypeError, ValueError):
        params = {}
    if "layer_type" in params:
        return rotary(hidden, position_ids, layer_type)
    return rotary(hidden, position_ids)


def _calibrate_with_cpu_fallback(model, calibration_tensor, **kwargs):
    """Run layer-wise calibration on the accelerator, falling back to CPU.

    The GPU path shuttles one layer at a time onto the card, which is what makes
    an 8B BF16 model calibrate inside 16 GB. When the linearized MoE experts
    refuse to move (see LinearizedExpertsWontMove), that path cannot produce
    correct activation scales, so fall back to calibrating on CPU rather than
    silently measuring a half-moved layer.

    CPU calibration is correct but slow - roughly 30 s per 1024-token sample -
    so this warns loudly rather than hiding the cost.
    """
    try:
        return calibrate_input_global_scales_layerwise(model, calibration_tensor, **kwargs)
    except LinearizedExpertsWontMove as exc:
        logger.warning("=" * 60)
        logger.warning("GPU layer-wise calibration unavailable: %s", exc)
        logger.warning(
            "Falling back to CPU calibration. This is CORRECT but SLOW (~30 s per "
            "1024-token sample). For a full run, budget hours, not minutes."
        )
        logger.warning("=" * 60)
        kwargs["device"] = "cpu"
        return calibrate_input_global_scales_layerwise(model, calibration_tensor, **kwargs)


def calibrate_input_global_scales_layerwise(
    model: Any,
    calibration_tensor: Any,
    batch_size: int = 1,
    num_samples: int | None = None,
    device: str = "cuda:0",
    max_layer: int | None = None,
    log_every_layer: int = 5,
    use_soar: bool = True,
    use_mr_gptq: bool = False,
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
    activation_cache: dict[str, torch.Tensor] = {}  # reservoir-sampled vectors
    block_maxes_store: dict[str, list[torch.Tensor]] = {}  # per-block maxes for SOAR
    hessian_store: dict[str, torch.Tensor] = {}  # per-Linear X^T X for GPTQ
    sample_counts: dict[str, int] = {}
    CACHE_SIZE = 128  # reservoir-sampled vectors per Linear (activation_cache)
    BLOCK_MAXES_CAP = 64  # max block-maxes tensors per Linear (SOAR); bounded to prevent OOM
    GROUP_SIZE = 16  # NVFP4 block size

    def _cache_sample(name: str, vec: torch.Tensor) -> None:
        """Reservoir sampling: keep up to CACHE_SIZE random vectors per key."""
        sample_counts[name] = sample_counts.get(name, 0) + 1
        n = sample_counts[name]
        if n <= CACHE_SIZE:
            if name not in activation_cache:
                activation_cache[name] = vec
            else:
                activation_cache[name] = torch.cat([activation_cache[name], vec], dim=0)
        else:
            # Replace random existing element with prob CACHE_SIZE / n
            if torch.rand(1).item() < CACHE_SIZE / n:
                idx = torch.randint(0, CACHE_SIZE, (1,)).item()
                activation_cache[name][idx : idx + 1] = vec

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
                xf = x.detach().float()
                cur_max = float(xf.abs().max().item())
            prev = activation_max.get(mod_name, 0.0)
            if cur_max > prev:
                activation_max[mod_name] = cur_max

            # Cache activation vectors and per-block maxes for SOAR optimization.
            # Skip when use_soar=False (e.g. stats-only pass) to avoid unbounded
            # memory growth: block_maxes accumulate one tensor per hook call per
            # module, growing to 40-60 GB on 977 calibration samples.
            if use_soar or use_mr_gptq:
                x2d = xf.reshape(-1, xf.shape[-1])
                nt = x2d.shape[0]
                sample_positions = [0]
                if nt > 2:
                    sample_positions.append(nt - 1)
                if nt > 8:
                    sample_positions.append(nt // 4)
                for pos in sample_positions:
                    _cache_sample(mod_name, x2d[pos : pos + 1].cpu())

            if use_soar:
                # Compute per-block maxes for SOAR scale optimization.
                # Capped at BLOCK_MAXES_CAP per module: unbounded accumulation across
                # all 977 samples × 1320 modules = ~41 GB (OOM). 64 samples is
                # sufficient for the SOAR grid-search MSE estimate.
                existing = block_maxes_store.get(mod_name)
                if existing is None or len(existing) < BLOCK_MAXES_CAP:
                    hidden = xf.shape[-1]
                    n_complete = (hidden // GROUP_SIZE) * GROUP_SIZE
                    blocks = xf[..., :n_complete].reshape(-1, GROUP_SIZE)
                    bmax = blocks.abs().max(dim=1).values.cpu()
                    # Cap bmax length to prevent OOM from popular experts that
                    # route many tokens (up to 512 tokens → [65536] tensor × 64
                    # entries × 1320 modules ≈ 21 GB). 1024 entries is enough
                    # for the SOAR grid-search MSE estimate.
                    if len(bmax) > 1024:
                        idx = torch.randperm(len(bmax))[:1024]
                        bmax = bmax[idx]
                    if existing is None:
                        block_maxes_store[mod_name] = []
                    block_maxes_store[mod_name].append(bmax)

            # Accumulate Hessian X^T X for GPTQ.
            # Hooks are registered per-layer (only current layer active), so at
            # most 1 layer's worth of Hessians (~1.3 GB) are on GPU at any time.
            # Keeping on GPU avoids the 31K D2H transfers per layer that cause
            # 5× slowdown when using .cpu() storage.
            if use_mr_gptq:
                x2d_gpu = x2d.to(xf.device)
                H_inc = x2d_gpu.T @ x2d_gpu
                if mod_name not in hessian_store:
                    hessian_store[mod_name] = H_inc
                else:
                    hessian_store[mod_name] = hessian_store[mod_name] + H_inc

        return _pre_hook

    if isinstance(calibration_tensor, (str, os.PathLike)):
        calibration_tensor = torch.load(calibration_tensor)

    n_total = calibration_tensor.shape[0]
    n = n_total if num_samples is None else min(num_samples, n_total)
    seq_len = calibration_tensor.shape[1]
    logger.info("Layer-wise calibration: %d/%d samples × %d tokens, batch=%d", n, n_total, seq_len, batch_size)

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
        # This routine computes the rotary output ONCE and shares it across every
        # layer. That is only valid while all layers have the same type - assert
        # it rather than silently sharing the wrong frequencies.
        _layer_types = list(getattr(model.config, "layer_types", None) or [])
        _unique_types = sorted(set(_layer_types))
        if len(_unique_types) > 1:
            raise NotImplementedError(
                f"config.layer_types contains multiple types {_unique_types}; this "
                "routine shares one rotary result across all layers, which is only "
                "correct for a single type. Compute per-type before proceeding."
            )
        _rope_layer_type = _unique_types[0] if _unique_types else None
        _rope_params = getattr(model.config, "rope_parameters", None) or {}
        _swa_layer_type = next(
            (t for t in _rope_params if "sliding" in str(t)), None
        )
        logger.info(
            "Rotary layer_type=%r (%d layers, %d distinct); swa layer_type=%r",
            _rope_layer_type,
            len(_layer_types),
            len(_unique_types),
            _swa_layer_type,
        )

        position_embeddings = (
            _call_rotary(rotary, dummy_h, position_ids, _rope_layer_type)
            if rotary is not None
            else None
        )
        swa_position_embeddings = (
            _call_rotary(swa_rotary, dummy_h, position_ids, _swa_layer_type)
            if swa_rotary is not None
            else None
        )
        del dummy_h

    embed.to("cpu")
    if rotary is not None:
        rotary.to("cpu")
    if swa_rotary is not None:
        swa_rotary.to("cpu")
    torch.cuda.empty_cache()

    # Per-sample state: (hidden, prev_router_hidden_states) — all CPU.
    # `prev_router_hidden_states` starts as None for every sample. The refactored
    # ZayaDecoderLayer keeps the residual stream internally, so it is no longer
    # threaded through this loop.
    router_cpu: list[Any] = [None] * len(hidden_states_cpu)

    # ── Iterate layers ─────────────────────────────────────────
    t_start = time.time()
    for layer_idx in range(max_layer):
        layer = layers[layer_idx]
        layer.to(dev)

        # `layer.to(dev)` deliberately does not reach the linearized expert tensors.
        #
        # llm-compressor's LinearExperts2D.from_experts_module ends with
        #
        #     offload_kwargs = get_cache_init_kwargs(experts)
        #     for module in self.modules():
        #         offload_module(module, **offload_kwargs)
        #
        # so every expert submodule is placed under compressed-tensors offloading.
        # Offloaded parameters are held in an OffloadCache and onloaded on access,
        # which is why `.to()`, `param.data = param.data.to(dev)` and even rebinding
        # via `setattr(mod, name, nn.Parameter(...))` all appear to succeed and then
        # read back on CPU. Measured: 48 expert weights per layer (16 experts x 3
        # projections) behave this way while the other ~28 parameters move normally.
        #
        # The supported way to run a forward over them is `align_modules`, which the
        # compressed-tensors docs describe as "onloading modules to a device, and
        # disabling onload and offload attempts triggered by forward calls. Used for
        # sequential onloading of layers" - exactly this loop's access pattern.
        # See the alignment context opened around the sample loop below.

        # Pick rotary embeddings (sliding-window vs full)
        if swa_position_embeddings is not None and swa_layers is not None:
            emb_to_use = position_embeddings if swa_layers[layer_idx] == 0 else swa_position_embeddings
        else:
            emb_to_use = position_embeddings

        # Register hooks on quantized Linears within this layer
        local_hooks = []
        for name, mod in layer.named_modules():
            if _is_quantized_linear(mod):
                full_name = f"model.layers.{layer_idx}.{name}" if name else f"model.layers.{layer_idx}"
                local_hooks.append(mod.register_forward_pre_hook(_make_pre_hook(full_name)))

        # Forward each sample through this layer, with (a) the offloaded expert
        # parameters onloaded to `dev` and (b) every expert receiving every token.
        #
        # Without moe_calibration_context, LinearExperts2D.forward routes each token
        # only to its top-1 expert, so any expert the router never selects during
        # calibration gets NO activation observations and its input_global_scale has
        # to be fabricated by the repair path. Measured on the 4-layer dry run:
        # only 146 of 200 modules fired and 54 were "repaired" - a fabricated
        # activation scale on 27% of the modules, which is precisely where W4A4
        # turns into silent quality loss.
        #
        # The context flips LinearExperts2D.forward to `expert(hidden_states)` for
        # every expert and then subsets the output, which the llm-compressor docs
        # describe as guaranteeing that all experts receive data during calibration
        # forward passes. It costs ~16x the expert FLOPs here (num_experts), and
        # that is the intended trade: calibration is once, wrong scales are forever.
        with (
            moe_calibration_context(),
            align_modules(list(layer.modules()), execution_device=torch.device(dev)),
            torch.no_grad(),
        ):
            # Verified inside the alignment context: outside it the offloaded
            # experts legitimately read back as CPU, so checking there would be
            # measuring the wrong thing. A stray tensor here would silently
            # calibrate activation scales against a partially-onloaded layer.
            _stray = [
                n for n, pp in layer.named_parameters()
                if pp.device.type != torch.device(dev).type
            ] + [
                n for n, bb in layer.named_buffers()
                if bb.device.type != torch.device(dev).type
            ]
            if _stray:
                raise LinearizedExpertsWontMove(
                    f"layer {layer_idx}: {len(_stray)} tensors are not on {dev} even "
                    f"inside align_modules (first few: {_stray[:5]})."
                )

            for sample_idx in range(len(hidden_states_cpu)):
                h = hidden_states_cpu[sample_idx].to(dev)
                pr = router_cpu[sample_idx]
                if pr is not None:
                    pr = pr.to(dev)

                # Refactored ZayaDecoderLayer signature (transformers >=5):
                #
                #   forward(hidden_states, prev_router_hidden_states=None,
                #           attention_mask=None, past_key_values=None,
                #           position_embeddings=None, **kwargs)
                #       -> (hidden_states, router_hidden_states_next)
                #
                # Two changes from the pre-refactor form this loop was written
                # against: `residual` is gone - the layer keeps the residual stream
                # internally - and the return is a 2-tuple, not (outputs, residual,
                # router). The old call passed the residual POSITIONALLY into what
                # is now the prev_router_hidden_states slot and then passed
                # prev_router_hidden_states again by keyword, which is what produced
                #   TypeError: ZayaDecoderLayer.forward() got multiple values for
                #   argument 'prev_router_hidden_states'
                # Extra kwargs (position_ids, cache_position, use_cache,
                # output_attentions, cca_mask) are dropped; they are not in the
                # signature and the rotary result is passed via position_embeddings.
                try:
                    h_new, pr_new = layer(
                        h,
                        prev_router_hidden_states=pr,
                        attention_mask=None,
                        past_key_values=None,
                        position_embeddings=emb_to_use,
                    )
                except torch.cuda.OutOfMemoryError as oom:
                    logger.error("OOM at layer %d sample %d: %s", layer_idx, sample_idx, oom)
                    raise

                hidden_states_cpu[sample_idx] = h_new.detach().cpu()
                router_cpu[sample_idx] = pr_new.detach().cpu() if pr_new is not None else None

                del h, pr, h_new, pr_new

        for hk in local_hooks:
            hk.remove()

        # ── MR-GPTQ: apply Hessian-weighted correction before offloading ──────
        # All quantized Linears in this layer are still on GPU here.  For each
        # one that has an accumulated Hessian (i.e., was activated at least once),
        # run GPTQ to correct its BF16 weights in-place.  NVFP4PackedCompressor
        # later packs these GPTQ-corrected BF16 values losslessly (every value is
        # already on the FP4 E2M1 grid after _gptq_correction).
        if use_mr_gptq:
            gptq_applied = 0
            gptq_skipped = 0
            for name, mod in layer.named_modules():
                if not _is_quantized_linear(mod):
                    continue
                full_name = f"model.layers.{layer_idx}.{name}" if name else f"model.layers.{layer_idx}"
                H_hess = hessian_store.pop(full_name, None)
                if H_hess is None or mod.weight is None:
                    gptq_skipped += 1
                    continue
                W = mod.weight.data.float()  # [out, in] on GPU
                W_corrected = _gptq_correction(W, H_hess, group_size=GROUP_SIZE)
                mod.weight.data = W_corrected.to(mod.weight.dtype)
                del H_hess, W, W_corrected
                gptq_applied += 1
            if gptq_applied > 0:
                logger.debug(
                    "  GPTQ layer %d: applied to %d Linears (%d skipped — no activations)",
                    layer_idx,
                    gptq_applied,
                    gptq_skipped,
                )

        # Per-layer memory telemetry. The 2026-09-03 full run ran at ~26 s/layer
        # through layer 30 and then collapsed to ~294 s/layer with the GPU pinned
        # near capacity. Two hypotheses were checked and BOTH failed: host swap was
        # untouched (684 KB), and align_modules does clear keep_onloaded_values on
        # exit. Cause still unknown - so log the numbers rather than theorise.
        if layer_idx % 5 == 0 or layer_idx >= max_layer - 3:
            logger.info(
                "  mem[layer %d]: cuda_alloc=%.2fGiB cuda_reserved=%.2fGiB host_rss=%.1fGiB",
                layer_idx,
                torch.cuda.memory_allocated() / 2**30,
                torch.cuda.memory_reserved() / 2**30,
                __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss / 2**20,
            )

        layer.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()
        # Return fragmented heap pages to OS (prevents RSS inflation from
        # repeated large tensor alloc/free cycles across 80 layers).
        try:
            import ctypes

            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        if (layer_idx + 1) % log_every_layer == 0 or layer_idx == max_layer - 1:
            elapsed = time.time() - t_start
            done = layer_idx + 1
            eta = elapsed * (max_layer - done) / max(done, 1)
            observed = sum(1 for v in activation_max.values() if v > 0)
            logger.info(
                "  layer %d/%d done (%.1fs elapsed, ETA %.0fs, %d hooks fired)", done, max_layer, elapsed, eta, observed
            )

    logger.info("Layer-wise forward complete in %.0fs", time.time() - t_start)
    if use_mr_gptq:
        remaining = len(hessian_store)
        if remaining:
            logger.warning("MR-GPTQ: %d Hessians not applied (modules outside layer range?)", remaining)
            hessian_store.clear()
        logger.info("MR-GPTQ: GPTQ corrections applied per-layer during calibration")

    # ── SOAR global-scale optimization ─────────────────────────
    # SOAR (arXiv:2605.12245): minimize FP8 block-scale rounding error rather
    # than just fitting max activation. Unlike the earlier MSE approach (which
    # clipped outlier channels and hurt ARC-Easy by -6%), SOAR never clips —
    # it searches for the global_scale that makes per-block FP8 scales most
    # precise. The search space is log-spaced around the max-abs baseline, so
    # max-abs is always a candidate and SOAR can only improve or match it.
    if use_soar:
        logger.info("SOAR scale optimization: grid-searching optimal global_scale per Linear...")
        soar_count = 0
        optimized_scales: dict[str, float] = {}
        for mod_name, max_act in activation_max.items():
            if max_act <= 0:
                continue
            bm_list = block_maxes_store.get(mod_name, [])
            opt = _compute_soar_global_scale(max_act, bm_list)
            base = GLOBAL_SCALE_NUM / max_act
            if abs(opt - base) / base > 0.001:  # only count non-trivial changes
                soar_count += 1
            optimized_scales[mod_name] = opt
        logger.info("SOAR: adjusted global_scale on %d/%d Linears", soar_count, len(activation_max))
    else:
        logger.info("SOAR disabled — using max-abs calibration")
        optimized_scales: dict[str, float] = {}

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
    set_scales: dict[str, float] = {}
    for name, mod in target_modules:
        max_act = activation_max.get(name, 0.0)
        if max_act <= 0:
            missing.append((name, mod))
            continue
        base_scale = GLOBAL_SCALE_NUM / max_act
        # Use MSE-optimized scale if available, else fall back to max-abs
        scale = optimized_scales.get(name, base_scale)
        mod.input_global_scale = _torch.nn.Parameter(_torch.tensor(scale, dtype=_torch.float32), requires_grad=False)
        set_scales[name] = scale
        set_count += 1

    # An uncalibrated module keeps input_global_scale = 0, which makes its
    # block_scale (igs * vec_max / 6) zero and produces NaN/garbage logits the
    # moment that expert is routed to at inference. That corruption is SILENT —
    # the run exits 0 and writes a correctly-sized checkpoint. It cost a full
    # debugging session on 2026-08-15 (checkpoint scored at chance level with
    # byte-identical weights). Repair inline instead of emitting a broken
    # checkpoint: borrow the median scale from calibrated peers in the same
    # layer with the same Linear type, widening the layer radius if needed.
    # Same fallback logic as scripts/fix_uncalibrated_igs.py, applied at the
    # source so a bad checkpoint can never be written in the first place.
    if missing:
        logger.warning(
            "Missing activation observations on %d/%d modules — repairing with "
            "per-layer median fallback (first few: %s)",
            len(missing),
            len(target_modules),
            [n for n, _ in missing[:5]],
        )
        peers: dict[tuple[int | None, str], list[float]] = {}
        for name, scale in set_scales.items():
            fc_type = "linear_fc1" if "linear_fc1" in name else ("linear_fc2" if "linear_fc2" in name else "other")
            peers.setdefault((_extract_layer_idx(name), fc_type), []).append(scale)

        repaired = 0
        for name, mod in missing:
            layer_idx = _extract_layer_idx(name)
            fc_type = "linear_fc1" if "linear_fc1" in name else ("linear_fc2" if "linear_fc2" in name else "other")
            candidates: list[float] = []
            # Widen the search radius until same-type peers are found.
            for radius in range(0, 9):
                for probe in {layer_idx, (layer_idx or 0) - radius, (layer_idx or 0) + radius}:
                    candidates.extend(peers.get((probe, fc_type), []))
                if candidates:
                    break
            if not candidates:  # last resort: any calibrated module of this type
                for (_, ft), vals in peers.items():
                    if ft == fc_type:
                        candidates.extend(vals)
            if not candidates:
                logger.error("Cannot repair input_global_scale for %s — no calibrated peers", name)
                continue
            fallback = float(statistics.median(candidates))
            mod.input_global_scale = _torch.nn.Parameter(
                _torch.tensor(fallback, dtype=_torch.float32), requires_grad=False
            )
            repaired += 1
        logger.warning("Repaired input_global_scale on %d/%d uncalibrated modules", repaired, len(missing))
        if repaired < len(missing):
            raise RuntimeError(
                f"{len(missing) - repaired} modules have no usable input_global_scale and could not be "
                "repaired. Writing this checkpoint would produce silent NaN corruption at inference. "
                "Check the calibration data covers every expert."
            )
        set_count += repaired

    logger.info(
        "Set input_global_scale on %d/%d Linears (%d repaired from uncalibrated)",
        set_count,
        len(target_modules),
        len(missing),
    )

    return activation_max


def _apply_weight_global_scale_convention(
    compressed_params: dict[str, dict[str, torch.Tensor]],
) -> int:
    """Transform weight_scale/weight_global_scale to CT convention.

    CT's NVFP4 contract requires:
        weight_global_scale = 2688 / max_abs(W)    [float32 scalar]
        weight_scale       = s_true * weight_global_scale  [per-group, fp8]

    The quantize pipeline stores raw s_true = w_max_group / 6 in weight_scale
    and never sets weight_global_scale. This function patches both to satisfy
    the convention, so the vLLM CUTLASS loader reads them correctly.

    Returns the number of modules patched.
    """
    patched = 0
    for comp in compressed_params.values():
        ws = comp.get("weight_scale")
        if ws is None:
            continue
        ws_f32 = ws.float()
        max_abs_w = FP4_E2M1_MAX * ws_f32.abs().max().item()
        if max_abs_w <= 0:
            continue
        wgs = GLOBAL_SCALE_NUM / max_abs_w
        comp["weight_global_scale"] = torch.tensor([wgs], dtype=torch.float32)
        # Rewrite weight_scale = s_true * wgs (clamped to FP8 range)
        new_ws = (ws_f32 * wgs).clamp(max=FP8_E4M3_MAX, min=-FP8_E4M3_MAX)
        comp["weight_scale"] = new_ws.to(torch.float8_e4m3fn)
        patched += 1
    return patched


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
    logger.info(
        "Model: %s | Output: %s | Mode: %s",
        args.model_id,
        args.output_dir,
        f"DRY RUN ({DRY_RUN_LAYERS_W4A4:d}L)" if args.dry_run else "FULL",
    )
    logger.info(
        "Calibration: %s | samples: %s | batch: %d",
        args.calibration_data,
        args.calibration_num_samples or "all",
        args.calibration_batch_size,
    )
    logger.info("Ignore patterns: %s", W4A4_IGNORE_PATTERNS)
    logger.info(
        "torch %s | CUDA: %s | %s",
        torch.__version__,
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    )
    logger.info("compressed_tensors %s", getattr(ct, "__version__", "?"))
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA required for W4A4 calibration forward pass")
        return 1

    cal_path = Path(args.calibration_data)
    if not cal_path.exists():
        logger.error("Calibration data not found: %s. Run scripts/build_calibration_data.py first.", cal_path)
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

    # Load with the MoE experts LINEARIZED.
    #
    # The refactored Zyphra/ZAYA1-8B stores experts as batched nn.Parameter on
    # ZayaExperts ([E, 2I, H] and [E, H, I]), applied via F.linear. Since
    # apply_quantization_config walks nn.Linear MODULES, it cannot see them: without
    # this, only 80 Linears (40 o_proj + 40 mlp.gate.down_proj) get quantized and the
    # entire MoE silently stays BF16, with no error raised. See RESEARCH.md 5.24.
    #
    # load_quantizable_moe patches from_pretrained to swap ZayaExperts for a
    # LinearExperts2D ModuleList of per-expert gate_proj/up_proj/down_proj Linears,
    # driven by the registry entries in scripts/register_zaya_moe.py.
    #
    # The checkpoint then SAVES in per-expert 2D form, and that is correct - it is
    # what vLLM's FusedMoE loader expects. RoutedExperts.build_expert_params_mapping()
    # maps checkpoint keys `experts.{i}.{gate_proj,up_proj,down_proj}.*` into the
    # fused w13/w2 parameters at load time. No re-fusing step is needed; the
    # `routed_experts.w13_*` names are vLLM's internal parameters, not checkpoint keys.
    from llmcompressor.modeling.moe.linearize import linearize_moe

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Convert-after-load: the batched weights are loaded correctly first, then each
    # ZayaExperts is replaced by a ModuleList of per-expert gate_proj/up_proj/
    # down_proj Linears with the weights COPIED across
    # (LinearExperts2D.from_experts_module). Slower than a mapping-based load
    # (2D->3D->2D) but correct; the mapping path needs converters that split the
    # batched tensors, which do not exist for zaya.
    _pre = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    linearize_moe(model)
    _post = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    logger.info("Linearized MoE experts: nn.Linear count %d -> %d", _pre, _post)
    if _post <= _pre:
        raise RuntimeError(
            f"linearize_moe did not expand the Linear count ({_pre} -> {_post}); "
            "the experts are still batched nn.Parameter and would be left in BF16. "
            "See RESEARCH.md 5.24 before proceeding."
        )
    logger.info(
        "Model loaded in %.0fs | Params: %.1fB", time.time() - t0, sum(p.numel() for p in model.parameters()) / 1e9
    )

    # Generation config sanity (matches W4A16 path)
    gc = model.generation_config
    if not gc.do_sample and gc.top_p is not None:
        gc.top_p = None
    if not gc.do_sample and gc.top_k is not None:
        gc.top_k = None

    # ── Apply NVFP4 (W4A4) scheme with ignore list ──────────
    scheme = preset_name_to_scheme("NVFP4", targets=["Linear"])
    logger.info(
        "Scheme: W=%s/g=%s, A=%s/g=%s (dyn=%s)",
        scheme.weights.num_bits,
        scheme.weights.group_size,
        scheme.input_activations.num_bits,
        scheme.input_activations.group_size,
        scheme.input_activations.dynamic,
    )

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
    logger.info("Restored plain Linear.forward on %d quantized Linears for BF16 calibration", restored)

    # ── Activation calibration ──────────────────────────────
    logger.info("Loading calibration tensor: %s", cal_path)
    cal_tensor = torch.load(cal_path)
    logger.info("Calibration shape: %s, dtype: %s", list(cal_tensor.shape), cal_tensor.dtype)

    if args.dry_run:
        # Keep the calibration cheap during dry-run, but honour an explicit
        # --calibration-num-samples so the dry run can be used to separate the
        # fixed per-layer cost from the per-sample cost when estimating a full
        # run. Two points at different sample counts give a defensible ETA;
        # one point does not.
        _dry_n = args.calibration_num_samples or 8
        cal_tensor = cal_tensor[: min(_dry_n, cal_tensor.shape[0])]
        logger.info("DRY RUN: truncated calibration to %d samples", cal_tensor.shape[0])

    # Layer-wise GPU calibration — full-model CPU forward crashes for Zaya
    # (CCA attention requires CUDA). 8B BF16 doesn't fit in 16 GB VRAM as a
    # whole, so we move one decoder layer at a time on/off GPU.
    max_layer = DRY_RUN_LAYERS_W4A4 if args.dry_run else None
    activation_max = _calibrate_with_cpu_fallback(
        model=model,
        calibration_tensor=cal_tensor,
        batch_size=args.calibration_batch_size,
        num_samples=args.calibration_num_samples,
        max_layer=max_layer,
        use_soar=not args.no_soar,
        use_mr_gptq=getattr(args, "mr_gptq", False),
    )

    # Diagnostic: distribution of input_global_scales
    if activation_max:
        import statistics as stats

        vals = sorted(activation_max.values())
        logger.info("Activation max distribution across Linears:")
        logger.info(
            "  min: %.4f | p25: %.4f | median: %.4f | p75: %.4f | max: %.4f",
            vals[0],
            vals[len(vals) // 4],
            stats.median(vals),
            vals[3 * len(vals) // 4],
            vals[-1],
        )

    # ── Stats-only early exit ──────────────────────────────────
    # Patch an existing checkpoint's manifest with activation_max_per_module
    # without re-running quantization. Useful when SOAR checkpoint already
    # exists but was quantized before this field was added.
    if getattr(args, "stats_only", False) and activation_max:
        output_dir = Path(args.output_dir)
        manifest_path = output_dir / "quantization_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as _mf:
                _manifest = json.load(_mf)
            _manifest["activation_max_per_module"] = {k: round(float(v), 4) for k, v in activation_max.items()}
            with open(manifest_path, "w") as _mf:
                json.dump(_manifest, _mf, indent=2)
            logger.info(
                "stats-only: wrote activation_max_per_module (%d entries) to %s",
                len(activation_max),
                manifest_path,
            )
        else:
            logger.error("stats-only: no quantization_manifest.json in %s", output_dir)
        return 0

    # ── Dynamic mixed-precision exemption ────────────────────
    # FusedMoE requires uniform quantization across ALL experts in a layer, so
    # when any single expert's activation exceeds the threshold we exempt the
    # ENTIRE layer's MLP. BF16 modules will be saved as raw weights in the
    # checkpoint and added to the quantization_config ignore list so vLLM routes
    # them through the standard BF16 matmul path.
    dynamic_bf16_set: set[str] = set()  # module names exempted from W4A4
    dynamic_outlier_layers: set[int] = set()  # layer indices driving the exemption
    threshold = getattr(args, "mixed_precision_threshold", MIXED_PRECISION_DEFAULT_THRESHOLD)
    if threshold is not None and threshold > 0:
        for name, max_act in activation_max.items():
            if max_act > threshold:
                layer_idx = _extract_layer_idx(name)
                if layer_idx is not None:
                    dynamic_outlier_layers.add(layer_idx)

        no_exempt = getattr(args, "no_bf16_exempt", False)
        if dynamic_outlier_layers and no_exempt:
            # Detection-only mode: record which layers are outliers in the
            # manifest, but compress them anyway. ARCQuant then fits an
            # additive residual correction for them at load time. Avoids the
            # ~3.5 GB cost of exempting 16 experts per layer to protect a few.
            logger.info(
                "Mixed precision: DETECTION ONLY (--no-bf16-exempt) - "
                "%d outlier layers recorded but compressed to W4A4",
                len(dynamic_outlier_layers),
            )
            logger.info("  Outlier layers: %s", sorted(dynamic_outlier_layers))
            logger.info("  Run build_arcquant_corrections.py against this checkpoint.")
        elif dynamic_outlier_layers:
            for name, mod in model.named_modules():
                if not _is_quantized_linear(mod):
                    continue
                layer_idx = _extract_layer_idx(name)
                if layer_idx in dynamic_outlier_layers:
                    dynamic_bf16_set.add(name)

            logger.info(
                "Mixed precision: %d outlier layers → %d modules exempted to BF16 (max_abs > %.1f)",
                len(dynamic_outlier_layers),
                len(dynamic_bf16_set),
                threshold,
            )
            logger.info("  Outlier layers: %s", sorted(dynamic_outlier_layers))
            # Log the worst offender per outlier layer
            for layer_idx in sorted(dynamic_outlier_layers):
                worst_name = max(
                    (n for n in activation_max if _extract_layer_idx(n) == layer_idx),
                    key=lambda n: activation_max[n],
                    default=None,
                )
                if worst_name:
                    logger.info(
                        "  L%d worst expert: max_abs=%.1f (%s)",
                        layer_idx,
                        activation_max[worst_name],
                        worst_name.split(".")[-1],
                    )
        else:
            logger.info(
                "Mixed precision: no layers exceeded threshold %.1f — all W4A4",
                threshold,
            )

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

    logger.info("Calibrated %d weight scales (%d layers skipped by dry-run)", calibrated_w, len(skipped_layers))

    # ── Compress each quantized module ──────────────────────
    compressed_params: dict[str, dict[str, torch.Tensor]] = {}
    linear_count = 0
    skipped = 0

    logger.info("Compressing weights with NVFP4PackedCompressor...")
    t0 = time.time()

    bf16_exempted_count = 0
    for name, module in model.named_modules():
        qscheme = getattr(module, "quantization_scheme", None)
        if qscheme is None or qscheme.weights is None:
            continue
        if args.dry_run and name in skipped_layers:
            continue

        # Mixed-precision exemption: entire outlier layers stay BF16.
        if name in dynamic_bf16_set:
            bf16_exempted_count += 1
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

    logger.info(
        "Compressed %d Linear modules in %.0fs (skipped %d, BF16-exempted %d)",
        linear_count,
        time.time() - t0,
        skipped,
        bf16_exempted_count,
    )

    # ── Apply CT weight_global_scale convention ─────────────
    wgs_patched = _apply_weight_global_scale_convention(compressed_params)
    logger.info("Applied weight_global_scale convention on %d modules", wgs_patched)

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

    # Quant-specific param suffixes that must NOT appear in the checkpoint for
    # BF16-exempted layers. The original Zyphra NVFP4 model stores these on
    # every Linear; we strip them so BF16 layers only carry their .weight tensor.
    # This makes the checkpoint unambiguous and eliminates the need for any
    # defensive guards in vLLM's load_weights path.
    _BF16_QUANT_SUFFIXES: frozenset[str] = frozenset(
        {
            "weight_scale",
            "weight_global_scale",
            "weight_zero_point",
            "input_global_scale",
            "weight_packed",
        }
    )

    for pname, param in model.named_parameters():
        module_name = ".".join(pname.split(".")[:-1])
        param_short = pname.split(".")[-1]

        if dryrun_cutoff is not None:
            li = _extract_layer_idx(pname)
            if li is not None and li >= dryrun_cutoff:
                dropped_dryrun += 1
                continue

        # BF16-exempted modules: drop all quant-specific tensors from the
        # original Zyphra NVFP4 format. Only the BF16 .weight survives.
        if module_name in dynamic_bf16_set and param_short in _BF16_QUANT_SUFFIXES:
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
        logger.info(
            "DRY RUN: dropped %d params from layers ≥%d (avoids ~16 GB BF16 bloat)", dropped_dryrun, dryrun_cutoff
        )

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

    if dynamic_bf16_set:
        logger.info(
            "Scrubbed quant tensors from %d BF16-exempted Linear modules — checkpoint contains only .weight for those layers",
            len(dynamic_bf16_set),
        )
    logger.info(
        "Output: %d params | packed: %.0f MB | weight_scales: %.0f MB | input_global_scale: %.1f KB | other: %.0f MB",
        len(output_state),
        packed_total / 1e6,
        scale_total / 1e6,
        igs_total / 1e3,
        other_total / 1e6,
    )

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

    # vLLM's AutoTokenizer path needs tokenizer files in the checkpoint dir.
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))

    # vLLM compressed-tensors loader keys off config.json's `quantization_config`
    # block; HF save_pretrained omits it, so we merge it in here.
    config_path = output_dir / "config.json"
    with open(config_path) as cf:
        config_dict = json.load(cf)
    # Build the effective ignore list: static patterns + dynamic per-layer
    # patterns for outlier MoE layers. vLLM reads this to determine which
    # modules use BF16 vs NVFP4 at inference time.
    effective_ignore: list[str] = list(W4A4_IGNORE_PATTERNS)
    # Keyed on dynamic_bf16_set (what was ACTUALLY exempted), not
    # dynamic_outlier_layers (what was merely detected). Under
    # --no-bf16-exempt the layers are compressed, so listing them as ignored
    # would make vLLM look for BF16 weights that are not in the checkpoint.
    if dynamic_bf16_set:
        layer_nums = "|".join(str(i) for i in sorted(dynamic_outlier_layers))
        # Match all submodules within the outlier layers' MLP block
        dynamic_ignore_pattern = f"re:model\\.layers\\.({layer_nums})\\..+"
        effective_ignore.append(dynamic_ignore_pattern)
        logger.info("Dynamic ignore pattern added: %s", dynamic_ignore_pattern)

    config_dict["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "tensor_group",
                    "group_size": 16,
                    "symmetric": True,
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "tensor_group",
                    "group_size": 16,
                    "symmetric": True,
                    "dynamic": False,
                },
                "targets": ["Linear"],
            }
        },
        "ignore": effective_ignore,
    }
    with open(config_path, "w") as cf:
        json.dump(config_dict, cf, indent=2)

    st.save_file(output_state, str(output_dir / "model.safetensors"))
    index: dict[str, Any] = {
        "metadata": {"total_size": len(output_state)},
        "weight_map": {name: "model.safetensors" for name in output_state},
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    # Quantization manifest
    total_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    manifest = {
        "model": "Zyphra/ZAYA1-8B-legacy",
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
            "linears_calibrated": sum(
                1 for n, _ in [(k, v) for k, v in compressed_params.items() if "input_global_scale" in v]
            ),
            "soar": not getattr(args, "no_soar", False),
            "mr_gptq": getattr(args, "mr_gptq", False),
        },
        "mixed_precision": {
            "threshold": threshold,
            "outlier_layers": sorted(dynamic_outlier_layers),
            "bf16_exempted_modules": len(dynamic_bf16_set),
            "w4a4_compressed_modules": linear_count,
            "arcquant_target": bool(dynamic_outlier_layers) and not dynamic_bf16_set,
            "description": (
                (
                    f"{len(dynamic_outlier_layers)} MoE layers where max_abs > {threshold:.0f} "
                    f"kept at BF16 ({len(dynamic_bf16_set)} Linears). "
                    f"FusedMoE requires uniform quantization per layer."
                )
                if dynamic_bf16_set
                else (
                    f"{len(dynamic_outlier_layers)} MoE layers where max_abs > {threshold:.0f} "
                    f"detected but COMPRESSED to W4A4 (--no-bf16-exempt). "
                    f"Requires ARCQuant residual corrections for accuracy."
                )
            )
            if dynamic_outlier_layers
            else "disabled",
        },
        # Per-module activation max values from calibration — used by
        # apply_singlequant_rotations.py for calibration-based channel selection.
        "activation_max_per_module": {k: round(float(v), 4) for k, v in activation_max.items()},
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

    logger.info("Saved in %.0fs | Total: %.0f MB (%.2f GB)", time.time() - t0, total_bytes / 1e6, total_bytes / 1e9)

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

    logger.info(
        "IGS coverage: %d/%d keys present (%.1f%%), %d nonzero (%.1f%%), %d near-zero garbage",
        len(igs_keys),
        expected_igs,
        coverage * 100,
        nonzero_igs,
        nonzero_coverage * 100,
        near_zero_igs,
    )
    if sample_vals:
        logger.info("Sample IGS values: %s", [f"{v:.4f}" for v in sample_vals])

    # Threshold: ≥95% of compressed Linears must have a non-zero input_global_scale.
    # Anything lower means calibration didn't reach those modules.
    coverage_threshold = 0.95
    if nonzero_coverage < coverage_threshold:
        logger.error(
            "QUALITY GATE FAILED: input_global_scale coverage %.1f%% < %.0f%% threshold.",
            nonzero_coverage * 100,
            coverage_threshold * 100,
        )
        logger.error(
            "This indicates calibration forward pass did not exercise most quantized "
            "Linears. Common causes: model on wrong device, layer-wise iteration "
            "stopped early, or hooks failed to register."
        )
        return 2

    logger.info("")
    logger.info("=" * 60)
    if args.dry_run:
        logger.info("W4A4 DRY RUN PASSED — pipeline verified on %d layers", DRY_RUN_LAYERS_W4A4)
        logger.info("Ready for full: python scripts/quantize_zaya_ct_nvfp4.py --scheme w4a4")
    else:
        logger.info("W4A4 COMPLETE — NVFP4 W4A4 ZAYA1-8B")
        logger.info("Output: %s (%.1f GB)", output_dir, total_bytes / 1e9)
        if dynamic_outlier_layers and bf16_exempted_count:
            logger.info(
                "Mixed precision: %d outlier layers (%s) kept at BF16 MLP",
                len(dynamic_outlier_layers),
                sorted(dynamic_outlier_layers),
            )
            logger.info("  W4A4 modules: %d | BF16 modules: %d", linear_count, bf16_exempted_count)
        elif dynamic_outlier_layers:
            logger.info(
                "Mixed precision: %d outlier layers (%s) COMPRESSED to W4A4 (--no-bf16-exempt)",
                len(dynamic_outlier_layers),
                sorted(dynamic_outlier_layers),
            )
            logger.info("  W4A4 modules: %d | BF16 modules: 0", linear_count)
            logger.warning(
                "  Checkpoint is UNCORRECTED - outlier layers are compressed with no "
                "mitigation. Run build_arcquant_corrections.py before evaluating quality."
            )
        logger.info("Next: smoke test with VLLM_NVFP4_GEMM_BACKEND=cutlass to force SM120 CUTLASS")
    logger.info("=" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1: ZAYA1-8B → NVFP4 compressed-tensors")
    parser.add_argument(
        "--scheme",
        choices=["w4a16", "w4a4"],
        default="w4a16",
        help="Quantization scheme: w4a16 (legacy SOTA) or w4a4 (Path B Phase 1)",
    )
    parser.add_argument("--model-id", default=os.environ.get("ZAYA_MODEL_ID", DEFAULT_MODEL))
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT} for w4a16, {DEFAULT_OUTPUT_W4A4} for w4a4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Quantize only first {DRY_RUN_LAYERS} (w4a16) / {DRY_RUN_LAYERS_W4A4} (w4a4) layers",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help=(
            "Run calibration and save activation_max_per_module to an existing checkpoint's "
            "quantization_manifest.json without re-quantizing. Useful for populating stats "
            "on an existing SOAR checkpoint for use by apply_singlequant_rotations.py."
        ),
    )
    parser.add_argument(
        "--calibration-data",
        default=DEFAULT_CALIBRATION_DATA,
        help=f"Path to calibration_data.pt (w4a4 only). Default: {DEFAULT_CALIBRATION_DATA}",
    )
    parser.add_argument(
        "--calibration-num-samples",
        type=int,
        default=None,
        help="Cap calibration to first N samples for quick iteration. Default: use the entire calibration tensor.",
    )
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=1,
        help="Batch size for calibration forward. Default 1 — 8B BF16 + 1024-token activations near saturate 16 GB.",
    )
    parser.add_argument(
        "--no-soar",
        action="store_true",
        default=False,
        help=(
            "Disable SOAR global-scale optimization (arXiv:2605.12245). "
            "By default SOAR is enabled: it grid-searches the optimal input_global_scale "
            "minimizing FP8 block-scale rounding error. Pass this flag to revert to "
            "plain max-abs calibration."
        ),
    )
    parser.add_argument(
        "--mr-gptq",
        action="store_true",
        default=False,
        dest="mr_gptq",
        help=(
            "Enable MR-GPTQ Hessian-weighted quantization (arXiv:2509.23202). "
            "During calibration, accumulates per-Linear X^T X Hessians. After each "
            "decoder layer's forward pass, applies the standard GPTQ column-by-column "
            "correction to the BF16 weights before they are packed to NVFP4. "
            "Requires the BF16 model (Zyphra/ZAYA1-8B), ~17 GB disk + ~1 GB extra VRAM "
            "per layer for Hessians. Adds ~30%% to calibration time. "
            "Expected accuracy gain: +2 to +4 pts ARC-Easy. Only applies to --scheme w4a4."
        ),
    )
    parser.add_argument(
        "--no-bf16-exempt",
        action="store_true",
        default=False,
        dest="no_bf16_exempt",
        help=(
            "Detect outlier layers and record them in the manifest, but compress "
            "them to W4A4 anyway instead of exempting them to BF16. Because FusedMoE "
            "requires uniform quantization per layer, normal exemption costs ~3.5 GB "
            "to protect ~24 offending Linears. Use this with "
            "build_arcquant_corrections.py, which fits an additive residual "
            "correction for the recorded outlier layers. Only applies to --scheme w4a4."
        ),
    )
    parser.add_argument(
        "--mixed-precision-threshold",
        type=float,
        default=MIXED_PRECISION_DEFAULT_THRESHOLD,
        metavar="MAX_ABS",
        dest="mixed_precision_threshold",
        help=(
            "W4A4 layers where any expert activation max_abs exceeds this value are kept "
            f"at BF16 instead of compressed (default: {MIXED_PRECISION_DEFAULT_THRESHOLD}). "
            "FusedMoE requires uniform quantization per layer, so the entire layer's MLP "
            "is exempted when any single expert is an outlier. "
            "Set to 0 to disable mixed precision. Only applies to --scheme w4a4."
        ),
    )
    args = parser.parse_args()

    # Register `zaya` with llm-compressor's batched-MoE linearizer, in THIS
    # process, before the pipeline builds. The refactored Zyphra/ZAYA1-8B
    # stores experts as nn.Parameter (ZayaExperts, modeling_zaya.py:576), not
    # nn.Linear, so targets:["Linear"] does not see them and the entire MoE
    # would silently stay BF16 with no error raised. See RESEARCH.md 5.24.
    sys.path.insert(0, str(Path(__file__).parent))
    from register_zaya_moe import register as _register_zaya_moe

    _register_zaya_moe()

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
    logger.info(
        "Model: %s | Output: %s | Mode: %s",
        args.model_id,
        args.output_dir,
        f"DRY RUN ({DRY_RUN_LAYERS:d}L)" if args.dry_run else "FULL",
    )
    logger.info("Compressor: NVFP4PackedCompressor (pack_fp4_to_uint8)")
    logger.info(
        "torch %s | CUDA: %s | %s",
        torch.__version__,
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    )
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
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Linearize the MoE experts here too. This path is weight-only, so it needs
    # none of the activation-calibration machinery, but it shares the one failure
    # that matters: ZayaExperts stores experts as batched nn.Parameter, and
    # apply_quantization_config walks nn.Linear MODULES. Without this, W4A16 would
    # quantize 80 Linears instead of 2000 and leave the entire MoE in BF16 with no
    # error raised - producing a barely-compressed checkpoint that looks fine.
    # Same bug as RESEARCH.md 5.24, different code path.
    from llmcompressor.modeling.moe.linearize import linearize_moe

    _pre = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    linearize_moe(model)
    _post = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    logger.info("Linearized MoE experts: nn.Linear count %d -> %d", _pre, _post)
    if _post <= _pre:
        raise RuntimeError(
            f"linearize_moe did not expand the Linear count ({_pre} -> {_post}); "
            "the experts are still batched nn.Parameter and would be left in BF16. "
            "See RESEARCH.md 5.24 before proceeding."
        )
    logger.info(
        "Model loaded in %.0fs | Params: %.1fB", time.time() - t0, sum(p.numel() for p in model.parameters()) / 1e9
    )

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
    logger.info(
        "Scheme: num_bits=%d, group_size=%d, strategy=%s",
        scheme.weights.num_bits,
        scheme.weights.group_size,
        scheme.weights.strategy,
    )

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

    # ── Apply CT weight_global_scale convention ─────────────
    wgs_patched = _apply_weight_global_scale_convention(compressed_params)
    logger.info("Applied weight_global_scale convention on %d modules", wgs_patched)

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

    logger.info(
        "Output: %d params | packed: %.0f MB | scales: %.0f MB | other: %.0f MB",
        len(output_state),
        packed_total / 1e6,
        scale_total / 1e6,
        other_total / 1e6,
    )

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
    total_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    manifest = {
        "model": "Zyphra/ZAYA1-8B-legacy",
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
    logger.info("Saved in %.0fs | Total: %.0f MB (%.2f GB)", elapsed, total_bytes / 1e6, total_bytes / 1e9)

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
