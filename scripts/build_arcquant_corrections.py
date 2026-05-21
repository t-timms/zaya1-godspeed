"""Build ARCQuant BF16 residual correction matrices for outlier W4A4 layers.

For each mixed-precision outlier layer that was kept at BF16, this script:
  1. Loads the W4A4 checkpoint and the BF16 reference model
  2. Runs calibration samples through the BF16 model to collect (input, output) pairs
     at each outlier ZayaBlock
  3. Identifies the top-k outlier input channels by max-abs activation
  4. Fits a linear residual: output_bf16 ≈ output_w4a4 + x_outlier @ arc_w
     where arc_w [n_outlier_ch, hidden_dim] is solved via least-squares
  5. Saves arc_residual_weight and arc_outlier_channels tensors into the
     W4A4 checkpoint directory so vLLM load_weights() picks them up

Usage:
  python3 scripts/build_arcquant_corrections.py \\
      --w4a4-checkpoint ./zaya1-8b-nvfp4-w4a4-soar \\
      --bf16-model Zyphra/ZAYA1-8B \\
      --calibration data/calibration/arcmix/calibration_data.pt \\
      --n-outlier-channels 32 \\
      --output-checkpoint ./zaya1-8b-nvfp4-w4a4-arc
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
HIDDEN_DIM = 2048
MAX_BF16_LAYERS_IN_VRAM = 2   # process this many BF16 layers simultaneously


# ── Weight loading helpers ───────────────────────────────────────────────────

def _load_safetensors_index(checkpoint_dir: Path) -> dict[str, str]:
    """Return {tensor_name: shard_filename} from model.safetensors.index.json."""
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    # single-shard
    shard_path = checkpoint_dir / "model.safetensors"
    if shard_path.exists():
        keys = list(load_file(str(shard_path), device="cpu").keys())
        return {k: "model.safetensors" for k in keys}
    raise FileNotFoundError(f"No safetensors shards found in {checkpoint_dir}")


def _load_layer_weights(
    weight_map: dict[str, str],
    checkpoint_dir: Path,
    layer_idx: int,
    prefix: str = "model.layers",
) -> dict[str, torch.Tensor]:
    """Load all tensors for one layer index from checkpoint shards (CPU)."""
    layer_prefix = f"{prefix}.{layer_idx}."
    relevant_shards: set[str] = set()
    for k, shard in weight_map.items():
        if k.startswith(layer_prefix):
            relevant_shards.add(shard)

    result: dict[str, torch.Tensor] = {}
    for shard in relevant_shards:
        shard_data = load_file(str(checkpoint_dir / shard), device="cpu")
        for k, v in shard_data.items():
            if k.startswith(layer_prefix):
                result[k] = v
    return result


# ── Simple BF16 MLP forward (for outlier layers) ────────────────────────────

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)).to(weight.dtype) * weight


def _swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """SwiGLU: silu(gate) * up, where gate_up is [tokens, 2*ffn_dim]."""
    half = gate_up.shape[-1] // 2
    gate, up = gate_up[:, :half], gate_up[:, half:]
    return nn.functional.silu(gate) * up


def _bf16_moe_forward_single_expert(
    hidden: torch.Tensor,         # [tokens_for_expert, hidden_dim] BF16
    fc1_weight: torch.Tensor,     # [ffn_dim*2, hidden_dim] BF16  (gate+up combined)
    fc2_weight: torch.Tensor,     # [hidden_dim, ffn_dim] BF16
) -> torch.Tensor:
    """Single-expert SwiGLU MLP forward, returns [tokens_for_expert, hidden_dim]."""
    gate_up = hidden.float() @ fc1_weight.float().T    # [T, 2*ffn]
    x = _swiglu(torch.tensor(gate_up, dtype=torch.bfloat16))   # [T, ffn]
    out = x.float() @ fc2_weight.float().T              # [T, hidden]
    return torch.tensor(out, dtype=torch.bfloat16)


def _bf16_layer_forward(
    hidden_states: torch.Tensor,          # [n_tokens, hidden_dim]
    layer_weights: dict[str, torch.Tensor],
    layer_idx: int,
    router_weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run one BF16 ZayaBlock forward (router + MoE).

    Returns the MoE output BEFORE residual addition — same signal that
    ZayaBlock.forward() returns.
    """
    prefix = f"model.layers.{layer_idx}.zaya_block"

    # RMSNorm on input (applied by ZayaDecoderMLPLayer, not ZayaBlock; included here
    # because calibration inputs come post-norm from the full model)
    # Note: we receive hidden_states already post-norm from hook intercept, so
    # no norm needed here.

    # ── Router forward ──────────────────────────────────────────────────────
    # Router MLP: dense1 → gelu → dense2 → softmax → top-1
    # Key names: model.layers.{n}.zaya_block.router.*
    rp = f"{prefix}.router"
    # ZayaRouter uses a 2-layer MLP: fc1 (hidden → router_hidden), fc2 (router_hidden → n_experts)
    # We use a simplified router: just apply the fc2 to get expert logits
    hidden_for_router = hidden_states.float()
    # Try to find router fc layers
    fc1_key = f"{rp}.dense_h_to_4h.weight"
    fc2_key = f"{rp}.dense_4h_to_h.weight"
    if fc1_key in router_weights and fc2_key in router_weights:
        rfc1 = router_weights[fc1_key].float()
        rfc2 = router_weights[fc2_key].float()
        r_hidden = torch.nn.functional.gelu(hidden_for_router @ rfc1.T)
        logits = r_hidden @ rfc2.T
    else:
        # fallback: direct linear
        for k in router_weights:
            if "weight" in k and rp in k:
                logits = hidden_for_router @ router_weights[k].float().T
                break
        else:
            raise KeyError(f"Cannot find router weights for layer {layer_idx}")

    # top-1 routing
    expert_ids = logits.argmax(dim=-1)  # [n_tokens]
    n_experts = logits.shape[-1]
    output = torch.zeros_like(hidden_states)

    for exp_id in range(n_experts):
        mask = expert_ids == exp_id
        if not mask.any():
            continue
        h_exp = hidden_states[mask]  # [t_exp, hidden]

        fc1_key = f"model.layers.{layer_idx}.zaya_block.experts.local_experts.{exp_id}.linear_fc1.weight"
        fc2_key = f"model.layers.{layer_idx}.zaya_block.experts.local_experts.{exp_id}.linear_fc2.weight"
        if fc1_key not in layer_weights or fc2_key not in layer_weights:
            continue  # BF16 exempted layer may not have these; skip
        fc1_w = layer_weights[fc1_key].to(torch.bfloat16)
        fc2_w = layer_weights[fc2_key].to(torch.bfloat16)
        out_exp = _bf16_moe_forward_single_expert(h_exp, fc1_w, fc2_w)
        output[mask] = out_exp

    return output


# ── W4A4 approximate forward ─────────────────────────────────────────────────

def _unpack_nvfp4_weight(
    weight_packed: torch.Tensor,   # uint8 [out, in//2]
    weight_scale: torch.Tensor,    # float8_e4m3fn [out, in//16]
    weight_global_scale: torch.Tensor,  # float32 scalar
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Dequantize NVFP4 packed weights to BF16. Returns [out, in]."""
    # FP4 E2M1 representable values (positive side)
    FP4_VALUES = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )

    wp = weight_packed.to(torch.int32)
    lo = wp & 0x0F                    # [out, in//2] low nibble
    hi = (wp >> 4) & 0x0F            # [out, in//2] high nibble

    def nibble_to_fp4(nibble: torch.Tensor) -> torch.Tensor:
        sign = (nibble >> 3).float() * (-2) + 1   # 1 if bit3=0, -1 if bit3=1
        mag_idx = nibble & 0x07
        mag = FP4_VALUES[mag_idx.flatten()].reshape(mag_idx.shape)
        return sign * mag

    lo_f = nibble_to_fp4(lo)   # [out, in//2]
    hi_f = nibble_to_fp4(hi)   # [out, in//2]

    # Interleave: even positions = lo nibble, odd = hi nibble
    W_fp4 = torch.zeros(out_features, in_features, dtype=torch.float32)
    W_fp4[:, 0::2] = lo_f
    W_fp4[:, 1::2] = hi_f

    # Apply per-group FP8 scale
    ws = weight_scale.to(torch.float32)   # [out, in//16]
    gs = float(weight_global_scale.item())
    # scale_real = ws / gs  (gs = 2688/max_abs, stored as divisor)
    ws_real = ws / gs                      # [out, in//16]
    ws_expanded = ws_real.repeat_interleave(16, dim=1)[:, :in_features]  # [out, in]
    W_dq = W_fp4 * ws_expanded

    return W_dq.to(torch.bfloat16)


def _w4a4_moe_forward_single_expert(
    hidden: torch.Tensor,           # [t, hidden_dim]
    layer_weights: dict[str, torch.Tensor],
    layer_idx: int,
    exp_id: int,
) -> torch.Tensor:
    pfx = f"model.layers.{layer_idx}.zaya_block.experts.local_experts.{exp_id}"
    fc1_packed = layer_weights.get(f"{pfx}.linear_fc1.weight_packed")
    fc1_scale = layer_weights.get(f"{pfx}.linear_fc1.weight_scale_fp8")
    fc1_gs = layer_weights.get(f"{pfx}.linear_fc1.weight_global_scale")
    fc2_packed = layer_weights.get(f"{pfx}.linear_fc2.weight_packed")
    fc2_scale = layer_weights.get(f"{pfx}.linear_fc2.weight_scale_fp8")
    fc2_gs = layer_weights.get(f"{pfx}.linear_fc2.weight_global_scale")

    if any(t is None for t in [fc1_packed, fc1_scale, fc1_gs, fc2_packed, fc2_scale, fc2_gs]):
        return None  # layer is BF16 — skip; caller handles

    out_fc1, in_fc1 = fc1_packed.shape[0], fc1_packed.shape[1] * 2
    out_fc2, in_fc2 = fc2_packed.shape[0], fc2_packed.shape[1] * 2
    fc1_w = _unpack_nvfp4_weight(fc1_packed, fc1_scale, fc1_gs, out_fc1, in_fc1)
    fc2_w = _unpack_nvfp4_weight(fc2_packed, fc2_scale, fc2_gs, out_fc2, in_fc2)
    return _bf16_moe_forward_single_expert(hidden, fc1_w, fc2_w)


# ── Residual fitting ─────────────────────────────────────────────────────────

def _fit_arc_correction(
    x_in: torch.Tensor,       # [N, hidden_dim] - ZayaBlock input activations
    y_ref: torch.Tensor,      # [N, hidden_dim] - BF16 MoE output
    y_w4a4: torch.Tensor,     # [N, hidden_dim] - W4A4 MoE output
    n_outlier_channels: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit: (y_ref - y_w4a4) ≈ x_in[:, outlier_ch] @ arc_w via least-squares.

    Returns:
        arc_outlier_channels: int64 [n_outlier_ch] — indices of selected input channels
        arc_residual_weight: bfloat16 [n_outlier_ch, hidden_dim]
    """
    residual = (y_ref - y_w4a4).float()      # [N, hidden]

    # Select channels by max-abs activation
    ch_maxabs = x_in.float().abs().amax(dim=0)  # [hidden_dim]
    n_outlier_channels = min(n_outlier_channels, ch_maxabs.numel())
    outlier_channels = ch_maxabs.topk(n_outlier_channels).indices.sort().values  # sorted

    X = x_in[:, outlier_channels].float().to(device)   # [N, n_ch]
    R = residual.to(device)                              # [N, hidden]

    N, n_ch = X.shape

    # Ridge least-squares: arc_w = (X^T X + λI)^{-1} X^T R
    # λ = 1e-4 * mean(diag(X^T X)) for numerical stability
    XtX = X.T @ X                             # [n_ch, n_ch]
    lam = 1e-4 * XtX.diagonal().mean().clamp(min=1e-6)
    XtX.diagonal().add_(lam)
    XtR = X.T @ R                             # [n_ch, hidden]
    try:
        arc_w = torch.linalg.solve(XtX, XtR)  # [n_ch, hidden]
    except (torch.linalg.LinAlgError, RuntimeError):
        logger.warning("linalg.solve failed, falling back to lstsq")
        arc_w = torch.linalg.lstsq(X, R, rcond=None).solution  # [n_ch, hidden]

    residual_before = residual.norm(dim=-1).mean().item()
    residual_after = (R - X @ arc_w).norm(dim=-1).mean().item()
    logger.info(
        "    Residual norm: %.4f → %.4f (%.1f%% reduction)",
        residual_before, residual_after,
        100.0 * (residual_before - residual_after) / max(residual_before, 1e-9),
    )

    return (
        outlier_channels.cpu().to(torch.int64),
        arc_w.cpu().to(torch.bfloat16),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w4a4-checkpoint", required=True,
                        help="Path to W4A4 SOAR checkpoint dir (contains quantization_manifest.json)")
    parser.add_argument("--bf16-model", default="Zyphra/ZAYA1-8B",
                        help="BF16 reference model (HF hub ID or local path)")
    parser.add_argument("--calibration", default="data/calibration/arcmix/calibration_data.pt",
                        help="Calibration tensor path (same as used for W4A4 quantization)")
    parser.add_argument("--n-outlier-channels", type=int, default=32,
                        help="Number of input channels to use for residual correction per layer")
    parser.add_argument("--output-checkpoint", required=True,
                        help="Output checkpoint dir (copy of W4A4 + arc correction tensors)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=256,
                        help="Max calibration samples to use for fitting (fewer = faster)")
    args = parser.parse_args()

    device = args.device
    w4a4_dir = Path(args.w4a4_checkpoint)
    out_dir = Path(args.output_checkpoint)
    t0 = time.time()

    # ── Load manifest ────────────────────────────────────────────────────────
    manifest_path = w4a4_dir / "quantization_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"quantization_manifest.json not found in {w4a4_dir}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    outlier_layers: list[int] = manifest.get("mixed_precision", {}).get("outlier_layers", [])
    if not outlier_layers:
        logger.info("No outlier layers found in manifest — nothing to do.")
        return
    logger.info("Outlier layers from manifest: %s", outlier_layers)

    # ── Copy checkpoint to output dir ────────────────────────────────────────
    if out_dir.exists() and out_dir != w4a4_dir:
        logger.info("Output dir exists, skipping copy (using existing)")
    elif out_dir != w4a4_dir:
        logger.info("Copying checkpoint %s → %s", w4a4_dir, out_dir)
        shutil.copytree(w4a4_dir, out_dir)
    else:
        logger.info("Writing arc tensors in-place to %s", w4a4_dir)

    # ── Load calibration data ────────────────────────────────────────────────
    cal_path = Path(args.calibration)
    logger.info("Loading calibration data from %s", cal_path)
    cal_tensor = torch.load(cal_path, weights_only=True)  # [N, seq_len]
    n_samples = min(args.max_samples, cal_tensor.shape[0])
    cal_tensor = cal_tensor[:n_samples].long()
    logger.info("Using %d calibration samples", n_samples)

    # ── Load W4A4 and BF16 weight maps ──────────────────────────────────────
    logger.info("Building weight maps...")
    w4a4_weight_map = _load_safetensors_index(w4a4_dir)
    bf16_dir = Path(args.bf16_model) if Path(args.bf16_model).exists() else None

    if bf16_dir is None:
        # Try HuggingFace cache
        try:
            from huggingface_hub import snapshot_download
            bf16_dir = Path(snapshot_download(args.bf16_model))
        except Exception as e:
            raise RuntimeError(
                f"BF16 model not found locally at {args.bf16_model} and "
                f"huggingface_hub download failed: {e}"
            )
    bf16_weight_map = _load_safetensors_index(bf16_dir)
    logger.info("BF16 model found at %s", bf16_dir)

    # ── Load embed_tokens for tokenization (needed for activation collection) ─
    # We'll run a lightweight embedding + block-level forward using hooks on vLLM
    # or a standalone transformer forward. Since we want to avoid loading vLLM here,
    # we build a minimal forward pass using raw weights.

    logger.info("Loading embedding weights...")
    embed_shards: set[str] = set()
    for k, s in bf16_weight_map.items():
        if "embed_tokens" in k or "norm" in k.split(".")[-2:]:
            embed_shards.add(s)

    embed_weights: dict[str, torch.Tensor] = {}
    for shard in embed_shards:
        shard_data = load_file(str(bf16_dir / shard), device="cpu")
        for k, v in shard_data.items():
            if "embed_tokens" in k:
                embed_weights[k] = v

    embed_weight = embed_weights.get("model.embed_tokens.weight")
    if embed_weight is None:
        raise KeyError("embed_tokens.weight not found in BF16 model")
    embed_weight = embed_weight.to(torch.bfloat16)
    logger.info("Embedding table: %s", embed_weight.shape)

    # ── Collect activations via vLLM (if available) or standalone ──────────
    # Prefer vLLM because it handles the full model correctly including CCA attention.
    # We attach hooks on ZayaBlock to capture (input, bf16_output, w4a4_output).

    try:
        import sys
        sys.path.insert(0, "/home/ttimm/vllm-src")
        import vllm  # noqa: F401
        use_vllm = True
        logger.info("vLLM available — using hook-based activation collection")
    except ImportError:
        use_vllm = False
        logger.warning("vLLM not found — using standalone forward (less accurate)")

    arc_corrections: dict[int, dict[str, Any]] = {}  # layer_idx → {channels, weight}

    if use_vllm:
        arc_corrections = _collect_via_vllm(
            outlier_layers=outlier_layers,
            w4a4_dir=w4a4_dir,
            bf16_dir=bf16_dir,
            cal_tensor=cal_tensor,
            n_outlier_channels=args.n_outlier_channels,
            device=device,
        )
    else:
        arc_corrections = _collect_standalone(
            outlier_layers=outlier_layers,
            w4a4_weight_map=w4a4_weight_map,
            w4a4_dir=w4a4_dir,
            bf16_weight_map=bf16_weight_map,
            bf16_dir=bf16_dir,
            embed_weight=embed_weight,
            cal_tensor=cal_tensor,
            n_outlier_channels=args.n_outlier_channels,
            device=device,
        )

    if not arc_corrections:
        logger.error("No corrections computed — check model paths and manifest")
        return

    # ── Save correction tensors into output checkpoint ───────────────────────
    logger.info("Saving arc correction tensors to %s", out_dir)
    arc_tensors: dict[str, torch.Tensor] = {}
    for layer_idx, corr in arc_corrections.items():
        arc_tensors[f"model.layers.{layer_idx}.zaya_block.arc_outlier_channels"] = corr["channels"]
        arc_tensors[f"model.layers.{layer_idx}.zaya_block.arc_residual_weight"] = corr["weight"]

    arc_shard_path = out_dir / "model_arc_corrections.safetensors"
    save_file(arc_tensors, str(arc_shard_path))
    logger.info("Saved %d correction tensors to %s", len(arc_tensors), arc_shard_path)

    # Update model.safetensors.index.json to include the new shard
    index_path = out_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        for k in arc_tensors:
            index["weight_map"][k] = "model_arc_corrections.safetensors"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        logger.info("Updated model.safetensors.index.json with %d arc keys", len(arc_tensors))

    # Update manifest
    manifest_out = out_dir / "quantization_manifest.json"
    with open(manifest_out) as f:
        manifest_data = json.load(f)
    manifest_data["arcquant"] = {
        "enabled": True,
        "n_outlier_channels": args.n_outlier_channels,
        "layers_corrected": sorted(arc_corrections.keys()),
        "max_samples": n_samples,
    }
    with open(manifest_out, "w") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info("Done in %.0fs", time.time() - t0)


def _collect_via_vllm(
    outlier_layers: list[int],
    w4a4_dir: Path,
    bf16_dir: Path,
    cal_tensor: torch.Tensor,
    n_outlier_channels: int,
    device: str,
) -> dict[int, dict[str, Any]]:
    """Use vLLM to run both W4A4 and BF16 models and collect ZayaBlock I/O."""
    import sys
    sys.path.insert(0, "/home/ttimm/vllm-src")
    import torch
    from vllm import LLM, SamplingParams  # noqa: F401

    corrections: dict[int, dict[str, Any]] = {}
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

    # Collect token id lists for vLLM
    prompts_token_ids = [ids.tolist() for ids in cal_tensor]

    # ── W4A4 model ──────────────────────────────────────────────────────────
    logger.info("Loading W4A4 model for activation capture...")
    llm_w4a4 = LLM(
        model=str(w4a4_dir),
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        moe_backend="cutlass",
        gpu_memory_utilization=0.4,
    )

    # Attach hooks via apply_model (the stable public API for model access).
    # Hooks persist in the worker across the generate() call.
    w4a4_inputs: dict[int, list[torch.Tensor]] = {li: [] for li in outlier_layers}
    w4a4_outputs: dict[int, list[torch.Tensor]] = {li: [] for li in outlier_layers}
    hook_handles: list = []

    def _attach_w4a4_hooks(model: torch.nn.Module) -> int:
        from vllm.model_executor.models.zaya import ZayaBlock
        count = 0
        for name, module in model.named_modules():
            if not isinstance(module, ZayaBlock):
                continue
            parts = name.split(".")
            try:
                li_idx = parts.index("layers")
                layer_n = int(parts[li_idx + 1])
            except (ValueError, IndexError):
                continue
            if layer_n not in outlier_layers:
                continue

            def make_hooks(ln: int):
                def pre_hook(mod, inp):
                    w4a4_inputs[ln].append(inp[0].detach().cpu())
                def post_hook(mod, inp, out):
                    w4a4_outputs[ln].append(out[0].detach().cpu())
                return pre_hook, post_hook

            pre_h, post_h = make_hooks(layer_n)
            hook_handles.append(module.register_forward_pre_hook(pre_h))
            hook_handles.append(module.register_forward_hook(post_h))
            count += 1
        return count

    n_hooked = llm_w4a4.apply_model(_attach_w4a4_hooks)[0]
    logger.info("  Attached hooks to %d ZayaBlock modules", n_hooked)

    logger.info("Running W4A4 inference on %d samples...", len(prompts_token_ids))
    llm_w4a4.generate(prompt_token_ids=prompts_token_ids, sampling_params=sampling_params)

    for h in hook_handles:
        h.remove()
    hook_handles.clear()

    # Stack activations
    w4a4_in_cat = {li: torch.cat(w4a4_inputs[li], dim=0) for li in outlier_layers if w4a4_inputs[li]}
    w4a4_out_cat = {li: torch.cat(w4a4_outputs[li], dim=0) for li in outlier_layers if w4a4_outputs[li]}

    del llm_w4a4
    torch.cuda.empty_cache()

    # ── BF16 model ──────────────────────────────────────────────────────────
    logger.info("Loading BF16 model for reference activations...")
    llm_bf16 = LLM(
        model=str(bf16_dir),
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
    )

    bf16_outputs: dict[int, list[torch.Tensor]] = {li: [] for li in outlier_layers}
    hook_handles_bf16: list = []

    def _attach_bf16_hooks(model: torch.nn.Module) -> int:
        from vllm.model_executor.models.zaya import ZayaBlock
        count = 0
        for name, module in model.named_modules():
            if not isinstance(module, ZayaBlock):
                continue
            parts = name.split(".")
            try:
                li_idx = parts.index("layers")
                layer_n = int(parts[li_idx + 1])
            except (ValueError, IndexError):
                continue
            if layer_n not in outlier_layers:
                continue

            def make_post_hook(ln: int):
                def post_hook(mod, inp, out):
                    bf16_outputs[ln].append(out[0].detach().cpu())
                return post_hook

            hook_handles_bf16.append(module.register_forward_hook(make_post_hook(layer_n)))
            count += 1
        return count

    n_hooked_bf16 = llm_bf16.apply_model(_attach_bf16_hooks)[0]
    logger.info("  Attached hooks to %d ZayaBlock modules (BF16)", n_hooked_bf16)

    logger.info("Running BF16 inference on %d samples...", len(prompts_token_ids))
    llm_bf16.generate(prompt_token_ids=prompts_token_ids, sampling_params=sampling_params)

    for h in hook_handles_bf16:
        h.remove()

    bf16_out_cat = {li: torch.cat(bf16_outputs[li], dim=0) for li in outlier_layers if bf16_outputs[li]}

    del llm_bf16
    torch.cuda.empty_cache()

    # ── Fit corrections ──────────────────────────────────────────────────────
    for li in outlier_layers:
        if li not in w4a4_in_cat or li not in w4a4_out_cat or li not in bf16_out_cat:
            logger.warning("Layer %d: missing activations, skipping", li)
            continue
        x_in = w4a4_in_cat[li]
        y_w4a4 = w4a4_out_cat[li]
        y_ref = bf16_out_cat[li]
        # Align shapes
        min_n = min(x_in.shape[0], y_w4a4.shape[0], y_ref.shape[0])
        x_in, y_w4a4, y_ref = x_in[:min_n], y_w4a4[:min_n], y_ref[:min_n]
        logger.info("Layer %d: fitting correction on %d token activations", li, min_n)
        channels, arc_w = _fit_arc_correction(x_in, y_ref, y_w4a4, n_outlier_channels, device)
        corrections[li] = {"channels": channels, "weight": arc_w}

    return corrections


def _collect_standalone(
    outlier_layers: list[int],
    w4a4_weight_map: dict[str, str],
    w4a4_dir: Path,
    bf16_weight_map: dict[str, str],
    bf16_dir: Path,
    embed_weight: torch.Tensor,
    cal_tensor: torch.Tensor,
    n_outlier_channels: int,
    device: str,
) -> dict[int, dict[str, Any]]:
    """Standalone (no-vLLM) activation collection using direct weight loading.

    Less accurate than the vLLM path because it skips attention layers and
    computes an approximate MoE forward using dequantized weights. Use only
    as a fallback.
    """
    corrections: dict[int, dict[str, Any]] = {}
    n_samples, seq_len = cal_tensor.shape

    # Embed all tokens
    logger.info("Embedding %d samples × %d tokens...", n_samples, seq_len)
    token_ids = cal_tensor.reshape(-1)  # [n_samples * seq_len]
    hidden_states = embed_weight[token_ids].to(torch.bfloat16)  # [N_total, hidden]

    # Load router weights once (small, fits in RAM)
    logger.info("Loading router weights for outlier layers...")
    router_weights: dict[str, torch.Tensor] = {}
    router_shards: set[str] = set()
    for k, s in bf16_weight_map.items():
        for li in outlier_layers:
            if f"layers.{li}.zaya_block.router" in k:
                router_shards.add(s)
                break
    for shard in router_shards:
        sd = load_file(str(bf16_dir / shard), device="cpu")
        for k, v in sd.items():
            for li in outlier_layers:
                if f"layers.{li}.zaya_block.router" in k:
                    router_weights[k] = v.to(torch.bfloat16)

    for layer_idx in sorted(outlier_layers):
        logger.info("Layer %d: loading weights...", layer_idx)
        t_layer = time.time()

        bf16_layer = _load_layer_weights(bf16_weight_map, bf16_dir, layer_idx)
        w4a4_layer = _load_layer_weights(w4a4_weight_map, w4a4_dir, layer_idx)

        # Convert BF16 weights
        bf16_layer = {k: v.to(torch.bfloat16) for k, v in bf16_layer.items()}

        logger.info("  Loaded layer weights in %.1fs", time.time() - t_layer)

        x_in_list: list[torch.Tensor] = []
        y_bf16_list: list[torch.Tensor] = []
        y_w4a4_list: list[torch.Tensor] = []

        # Process in chunks for memory efficiency
        chunk = 512
        for start in range(0, hidden_states.shape[0], chunk):
            h_chunk = hidden_states[start:start + chunk]
            x_in_list.append(h_chunk.cpu())

            # BF16 forward
            y_bf16 = _bf16_layer_forward(h_chunk, bf16_layer, layer_idx, router_weights)
            y_bf16_list.append(y_bf16.cpu())

            # W4A4 forward (dequantized approximation)
            logits = h_chunk.float() @ router_weights.get(
                f"model.layers.{layer_idx}.zaya_block.router.dense_h_to_4h.weight",
                torch.zeros(64, HIDDEN_DIM)
            ).float().T
            expert_ids = logits.argmax(dim=-1) if logits.shape[-1] > 1 else torch.zeros(h_chunk.shape[0], dtype=torch.long)
            n_experts_layer = max(expert_ids.max().item() + 1, 1)
            y_w4a4_chunk = torch.zeros_like(h_chunk)
            for exp_id in range(int(n_experts_layer)):
                mask = expert_ids == exp_id
                if not mask.any():
                    continue
                h_exp = h_chunk[mask]
                result = _w4a4_moe_forward_single_expert(h_exp, w4a4_layer, layer_idx, exp_id)
                if result is not None:
                    y_w4a4_chunk[mask] = result
                else:
                    # BF16 layer — use BF16 output directly (no correction needed)
                    y_w4a4_chunk[mask] = y_bf16[mask]
            y_w4a4_list.append(y_w4a4_chunk.cpu())

        x_in = torch.cat(x_in_list, dim=0)
        y_ref = torch.cat(y_bf16_list, dim=0)
        y_w4a4 = torch.cat(y_w4a4_list, dim=0)

        logger.info("  Fitting correction on %d tokens...", x_in.shape[0])
        channels, arc_w = _fit_arc_correction(x_in, y_ref, y_w4a4, n_outlier_channels, device)
        corrections[layer_idx] = {"channels": channels, "weight": arc_w}

        del bf16_layer, w4a4_layer, x_in_list, y_bf16_list, y_w4a4_list
        logger.info("  Layer %d done (%.0fs)", layer_idx, time.time() - t_layer)

    return corrections


if __name__ == "__main__":
    main()
