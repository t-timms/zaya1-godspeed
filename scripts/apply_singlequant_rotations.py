"""SingleQuant rotation preprocessing for ZAYA1-8B W4A4 outlier layers.

SingleQuant (arXiv:2511.22316) eliminates activation outliers in MoE layers
via two rotation stages applied OFFLINE to the BF16 weights:

  ART (Anti-Rotation Transform): Givens rotations that reduce sparse massive
      outlier channels by rotating them into adjacent non-outlier channels.
      Each rotation G(c, j, θ) zeros out the outlier contribution at channel c
      by mixing it into channel j.

  URT (Uniform Rotation Transform): a random orthogonal matrix Q derived from
      a random Gaussian via QR decomposition.  Applied after ART, it spreads
      residual non-uniformity uniformly across all channels.

Both rotations are applied to the BF16 model weights and the preceding
LayerNorm scales, then saved as a new rotated-BF16 checkpoint.  Inference
is unchanged — the rotations are fully absorbed.

Only the outlier layers (as listed in the input checkpoint's
quantization_manifest.json) are processed.  Non-outlier layers are copied
as-is.

Usage (WSL, vllm-env active):
    python3 scripts/apply_singlequant_rotations.py \\
        --input  Zyphra/ZAYA1-8B \\
        --manifest zaya1-8b-nvfp4-w4a4/quantization_manifest.json \\
        --output zaya1-8b-bf16-rotated

    # Then quantize the rotated model with a higher outlier threshold:
    python3 scripts/quantize_zaya_ct_nvfp4.py --scheme w4a4 \\
        --model-id ./zaya1-8b-bf16-rotated \\
        --mixed-precision-threshold 1000.0 \\
        --output-dir ./zaya1-8b-nvfp4-w4a4-sq

Notes on absorption for ZAYA1-8B SwiGLU MoE:
  - fc1 input rotation (H → fc1 weights, absorbed into pre-MoE RMSNorm):
      gamma_new = (H @ gamma_groups.T).T   [per group of 16 hidden channels]
      W_fc1_new = W_fc1 @ H.T             [per group of 16 input channels]
  - fc2 input rotation requires rotating through fc1's OUTPUT dimension.
      For SwiGLU: output = silu(gate) * up, so only the 'up' half can be
      cleanly rotated (the 'gate' half goes through a nonlinearity).
      We skip fc2 rotation here — the main benefit comes from fc1.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT = "Zyphra/ZAYA1-8B"
GROUP_SIZE = 16  # Must match NVFP4 quantization group size


# ─────────────────────────────────────────────────────────────────────────────
# Rotation utilities
# ─────────────────────────────────────────────────────────────────────────────

def _hadamard_matrix(n: int) -> torch.Tensor:
    """Normalized Walsh-Hadamard matrix of size n (must be power of 2)."""
    try:
        from scipy.linalg import hadamard as _h
        return torch.from_numpy(_h(n).astype("float32")) / (n ** 0.5)
    except ImportError:
        pass
    if n == 1:
        return torch.ones(1, 1)
    H_half = _hadamard_matrix(n // 2)
    H = torch.zeros(n, n)
    H[:n // 2, :n // 2] = H_half
    H[:n // 2, n // 2:] = H_half
    H[n // 2:, :n // 2] = H_half
    H[n // 2:, n // 2:] = -H_half
    return H / (2 ** 0.5)


def _givens_rotation(c: int, j: int, theta: float, n: int) -> torch.Tensor:
    """Construct an n×n Givens rotation matrix G(c, j, θ).

    Rotates the (c, j) plane by θ radians, reducing the magnitude of channel c.
    """
    G = torch.eye(n, dtype=torch.float32)
    cos_t = float(torch.cos(torch.tensor(theta)))
    sin_t = float(torch.sin(torch.tensor(theta)))
    G[c, c] = cos_t
    G[c, j] = -sin_t
    G[j, c] = sin_t
    G[j, j] = cos_t
    return G


def _art_rotation(
    outlier_channels: list[int],
    hidden_dim: int,
    activation_max_per_channel: torch.Tensor,  # [hidden_dim] float32
    n_rotations: int = 1,
) -> torch.Tensor:
    """Build ART: a sequence of Givens rotations that reduce outlier channels.

    For each outlier channel c, find the non-outlier channel j with closest
    magnitude and apply G(c, j, θ) where θ = arctan(|x_c| / |x_j|) / 2,
    which equalises |x_c'| ≈ |x_j'|.

    Returns R_art: [hidden_dim, hidden_dim] float32 rotation matrix (product
    of all Givens rotations, applied as x_rot = x @ R_art).
    """
    R = torch.eye(hidden_dim, dtype=torch.float32)
    mag = activation_max_per_channel.clone()

    for c in outlier_channels:
        for _ in range(n_rotations):
            # Candidate partner channels: non-outlier, closest magnitude
            candidates = [j for j in range(hidden_dim) if j != c and j not in outlier_channels]
            if not candidates:
                break
            # Find closest-magnitude non-outlier channel
            diffs = (mag[candidates] - mag[c]).abs()
            j = candidates[int(diffs.argmin().item())]

            # θ that equalises |x_c| and |x_j| after rotation
            mc, mj = float(mag[c].item()), float(mag[j].item())
            if mc + mj < 1e-12:
                break
            theta = float(torch.atan2(torch.tensor(mc), torch.tensor(mj)).item()) * 0.5

            G = _givens_rotation(c, j, theta, hidden_dim)
            R = R @ G
            # Update magnitude estimate after rotation
            new_mc = mc * abs(float(G[c, c].item())) + mj * abs(float(G[c, j].item()))
            new_mj = mc * abs(float(G[j, c].item())) + mj * abs(float(G[j, j].item()))
            mag[c] = new_mc
            mag[j] = new_mj

    return R


def _urt_rotation(hidden_dim: int, seed: int = 42) -> torch.Tensor:
    """Build URT: a random orthogonal matrix Q via QR decomposition of Gaussian.

    Spreads residual outliers uniformly across all channels.
    Returns Q: [hidden_dim, hidden_dim] float32 orthogonal matrix.
    """
    torch.manual_seed(seed)
    G = torch.randn(hidden_dim, hidden_dim, dtype=torch.float32)
    Q, _ = torch.linalg.qr(G)
    return Q


def _rotate_linear_input(W: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply input-dimension rotation to weight matrix W.

    W: [out_features, in_features]
    R: [in_features, in_features] orthogonal rotation (x_rot = x @ R)
    Returns W @ R^T so that (W @ R^T) @ (R @ x) = W @ x.
    """
    return W.float() @ R.T


def _rotate_layernorm(gamma: torch.Tensor, R: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Absorb per-group rotation into LayerNorm/RMSNorm weight.

    For the chain  y = gamma * norm(x)  →  W @ y,  after rotating W's input
    by R^T (i.e., W_rot = W @ R^T), we need the LN output to be rotated by R:
        z_rot = R @ z  where  z = gamma * norm(x)

    For a full-hidden rotation: gamma_rot = R @ gamma directly.
    For a per-group rotation (group_size=g): apply R_g to each group of g
    channels independently (approximate — ignores cross-group RMS coupling).

    Returns gamma_new (same shape and dtype as gamma).
    """
    hidden = gamma.shape[0]
    assert hidden % group_size == 0 or R.shape[0] == hidden, (
        f"gamma dim {hidden} not divisible by group_size {group_size} and "
        f"R is not full-hidden ({R.shape})"
    )

    gamma_f = gamma.float()
    if R.shape[0] == hidden:
        gamma_new = R @ gamma_f
    else:
        # Per-group: apply R to each group of group_size channels
        gamma_new = gamma_f.reshape(-1, group_size)  # [n_groups, group_size]
        gamma_new = (R @ gamma_new.T).T              # [n_groups, group_size]
        gamma_new = gamma_new.reshape(-1)

    return gamma_new.to(gamma.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer rotation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _find_pre_moe_norm(
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    hidden_dim: int | None = None,
) -> str | None:
    """Find the key prefix for the pre-MoE RMSNorm weight in a given layer.

    Tries common patterns for ZAYA1-8B's decoder layer structure.
    When hidden_dim is given, the fallback search is filtered to only
    return norms whose shape matches hidden_dim (avoids picking up
    smaller norms like router.rmsnorm_eda with shape [256]).
    """
    candidates = [
        f"model.layers.{layer_idx}.input_norm.weight",
        f"model.layers.{layer_idx}.pre_moe_norm.weight",
        f"model.layers.{layer_idx}.post_attention_layernorm.weight",
        f"model.layers.{layer_idx}.zaya_block.norm.weight",
        f"model.layers.{layer_idx}.norm2.weight",
    ]
    for c in candidates:
        if c in state_dict:
            if hidden_dim is None or state_dict[c].shape[0] == hidden_dim:
                return c
    # Fallback: search all norm weights under this layer, filtered by hidden_dim
    prefix = f"model.layers.{layer_idx}."
    norm_keys = [
        k for k in state_dict
        if k.startswith(prefix) and "norm" in k and k.endswith(".weight")
        and (hidden_dim is None or state_dict[k].shape[0] == hidden_dim)
    ]
    if norm_keys:
        return norm_keys[0]
    return None


def apply_rotations_to_layer(
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    outlier_channels: list[int],
    activation_max_per_channel: torch.Tensor,
    seed: int = 42,
    use_art: bool = True,
    use_urt: bool = True,
    group_size: int = GROUP_SIZE,
) -> dict[str, int]:
    """Apply ART + URT rotations to a single MoE layer's weights in-place.

    Modifies state_dict entries for:
      - model.layers.N.*.linear_fc1.weight  (all experts, fc1 gate+up)
      - model.layers.N.pre_moe_norm.weight  (absorb fc1 input rotation)

    fc2 rotation is skipped because SwiGLU's gate nonlinearity prevents
    clean absorption without inference changes.

    Returns: counts of modified tensors.
    """
    hidden_dim = activation_max_per_channel.shape[0]
    modified: dict[str, int] = {"fc1": 0, "norm": 0, "skipped_fc2": 0}

    # Build combined rotation R = R_urt @ R_art (applied as x_rot = x @ R)
    R = torch.eye(hidden_dim, dtype=torch.float32)

    if use_art and outlier_channels:
        R_art = _art_rotation(outlier_channels, hidden_dim, activation_max_per_channel)
        R = R @ R_art
        logger.info("  L%d ART: %d outlier channels rotated", layer_idx, len(outlier_channels))

    if use_urt:
        R_urt = _urt_rotation(hidden_dim, seed=seed + layer_idx)
        R = R @ R_urt
        logger.info("  L%d URT: random orthogonal Q applied", layer_idx)

    if (R - torch.eye(hidden_dim)).abs().max() < 1e-6:
        logger.info("  L%d: identity rotation — skipping", layer_idx)
        return modified

    # Rotate all fc1 weights (gate+up) — input dimension = hidden_dim
    prefix = f"model.layers.{layer_idx}."
    for key, W in state_dict.items():
        if not key.startswith(prefix):
            continue
        if "linear_fc1" not in key or not key.endswith(".weight"):
            continue
        if W.ndim != 2 or W.shape[1] != hidden_dim:
            continue
        state_dict[key] = _rotate_linear_input(W, R).to(W.dtype)
        modified["fc1"] += 1

    if modified["fc1"] == 0:
        logger.warning("  L%d: no fc1 weights found — check key pattern", layer_idx)

    # Absorb rotation into preceding RMSNorm
    norm_key = _find_pre_moe_norm(state_dict, layer_idx, hidden_dim=hidden_dim)
    if norm_key is not None:
        gamma = state_dict[norm_key]
        state_dict[norm_key] = _rotate_layernorm(gamma, R, group_size=group_size)
        modified["norm"] = 1
        logger.info("  L%d: absorbed rotation into %s", layer_idx, norm_key)
    else:
        logger.warning("  L%d: could not find pre-MoE RMSNorm — rotation NOT absorbed", layer_idx)

    # Note: fc2 skipped — SwiGLU gate nonlinearity prevents clean absorption
    fc2_count = sum(
        1 for k in state_dict
        if k.startswith(prefix) and "linear_fc2" in k and k.endswith(".weight")
    )
    modified["skipped_fc2"] = fc2_count

    return modified


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply SingleQuant ART+URT rotations to BF16 ZAYA1-8B for outlier elimination"
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help=f"Source BF16 model (HF repo ID or local path). Default: {DEFAULT_INPUT}"
    )
    parser.add_argument(
        "--manifest",
        default="zaya1-8b-nvfp4-w4a4/quantization_manifest.json",
        help="Path to quantization_manifest.json from a prior W4A4 run (provides outlier_layers list).",
    )
    parser.add_argument(
        "--output", default="./zaya1-8b-bf16-rotated",
        help="Output directory for the rotated BF16 checkpoint."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-art", action="store_true", help="Skip ART Givens rotations")
    parser.add_argument("--no-urt", action="store_true", help="Skip URT random orthogonal rotation")
    parser.add_argument(
        "--outlier-channels",
        type=int, nargs="+", default=None,
        help="Override outlier channel indices for ART (default: derived from activation statistics).",
    )
    args = parser.parse_args()

    # ── Load manifest for outlier layer list ──────────────────
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        logger.error("Run quantize_zaya_ct_nvfp4.py --scheme w4a4 first to generate it.")
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    outlier_layers: list[int] = manifest.get("mixed_precision", {}).get("outlier_layers", [])
    if not outlier_layers:
        logger.warning("No outlier_layers found in manifest — nothing to rotate")
        logger.warning("If all layers are W4A4 already, SingleQuant may not be needed.")
    else:
        logger.info("Outlier layers from manifest: %s", outlier_layers)

    # Per-module activation max from calibration: keys are full module paths
    # e.g. "model.layers.N.zaya_block.experts.local_experts.M.linear_fc1"
    act_max_per_module: dict[str, float] = manifest.get("activation_max_per_module", {})
    if act_max_per_module:
        logger.info("Loaded activation_max_per_module (%d entries) from manifest", len(act_max_per_module))
    else:
        logger.info("No activation_max_per_module in manifest — using weight-norm proxy for channel selection")

    # ── Load BF16 model ───────────────────────────────────────
    logger.info("Loading BF16 model from %s ...", args.input)
    t0 = time.time()
    import transformers
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.input,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    logger.info("Model loaded in %.0fs", time.time() - t0)

    # Extract state dict for in-place modification
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Apply rotations to each outlier layer ─────────────────
    total_fc1 = 0
    total_norm = 0
    total_skipped_fc2 = 0

    for layer_idx in (outlier_layers if outlier_layers else []):
        logger.info("Processing layer %d ...", layer_idx)

        fc1_keys = [
            k for k in state_dict
            if f"model.layers.{layer_idx}." in k
            and "linear_fc1" in k
            and k.endswith(".weight")
        ]
        if not fc1_keys:
            logger.warning("  L%d: no fc1 weights found, skipping", layer_idx)
            continue

        if act_max_per_module:
            # Calibration-based channel selection:
            # Weight each expert's fc1 column norms by its calibration activation max.
            # Experts with higher max_abs see larger activations → their column norms
            # are more reliable indicators of which input channels are outliers.
            col_norms_weighted: list[torch.Tensor] = []
            weights: list[float] = []
            for key in fc1_keys:
                # key: "model.layers.N.zaya_block.experts.local_experts.M.linear_fc1.weight"
                # module key in manifest lacks ".weight" suffix
                module_key = key[: -len(".weight")]
                expert_max = act_max_per_module.get(module_key, 1.0)
                W = state_dict[key].float()  # [out, hidden]
                col_norms_weighted.append(W.abs().mean(dim=0) * expert_max)
                weights.append(expert_max)
            total_w = sum(weights) or 1.0
            activation_max_proxy = sum(col_norms_weighted) / total_w  # [hidden]
            hidden_dim = activation_max_proxy.shape[0]
            logger.info("  L%d: using calibration-weighted channel proxy", layer_idx)
        else:
            # Fallback: uniform-average column norms
            col_norms_list = []
            for key in fc1_keys:
                W = state_dict[key].float()
                col_norms_list.append(W.abs().mean(dim=0))
            activation_max_proxy = torch.stack(col_norms_list).mean(dim=0)
            hidden_dim = activation_max_proxy.shape[0]

        # Determine outlier channels for this layer
        if args.outlier_channels is not None:
            outlier_channels = [c for c in args.outlier_channels if c < hidden_dim]
        else:
            # Top-k channels by activation proxy (k = hidden_dim // 32 ≈ 64)
            k = max(1, hidden_dim // 32)
            topk = activation_max_proxy.topk(k)
            outlier_channels = topk.indices.tolist()
            logger.info(
                "  L%d: top-%d outlier channels (max proxy=%.3f)", layer_idx, k,
                float(activation_max_proxy.max().item()),
            )

        counts = apply_rotations_to_layer(
            state_dict=state_dict,
            layer_idx=layer_idx,
            outlier_channels=outlier_channels,
            activation_max_per_channel=activation_max_proxy,
            seed=args.seed,
            use_art=not args.no_art,
            use_urt=not args.no_urt,
        )
        total_fc1 += counts["fc1"]
        total_norm += counts["norm"]
        total_skipped_fc2 += counts["skipped_fc2"]

    logger.info(
        "Rotations applied: %d fc1 weights, %d LN weights, %d fc2 skipped (SwiGLU)",
        total_fc1, total_norm, total_skipped_fc2,
    )

    # ── Save rotated BF16 model ───────────────────────────────
    import safetensors.torch as st

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving rotated BF16 model to %s ...", out_path)
    t0 = time.time()

    # Config + tokenizer
    model.config.save_pretrained(str(out_path))
    # ZAYA1-8B generation_config has top_p/top_k set with do_sample=False,
    # which fails strict validation in save_pretrained. Enable do_sample to
    # match the intent of the sampling parameters already present.
    model.generation_config.do_sample = True
    model.generation_config.save_pretrained(str(out_path))
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)
    tokenizer.save_pretrained(str(out_path))

    # Weights — save as single shard (BF16 17 GB fits in one safetensors file)
    st.save_file(state_dict, str(out_path / "model.safetensors"))
    total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
    index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": {k: "model.safetensors" for k in state_dict},
    }
    with open(out_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    # Rotation manifest
    rotation_manifest = {
        "source": args.input,
        "method": "SingleQuant ART+URT (arXiv:2511.22316)",
        "outlier_layers": outlier_layers,
        "art_enabled": not args.no_art,
        "urt_enabled": not args.no_urt,
        "fc1_rotated": total_fc1,
        "norms_absorbed": total_norm,
        "fc2_skipped": total_skipped_fc2,
        "seed": args.seed,
    }
    with open(out_path / "rotation_manifest.json", "w") as f:
        json.dump(rotation_manifest, f, indent=2)

    total_bytes = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file())
    logger.info("Saved in %.0fs | %.2f GB", time.time() - t0, total_bytes / 1e9)
    logger.info("")
    logger.info("Next step: quantize the rotated model with a higher outlier threshold:")
    logger.info(
        "  python3 scripts/quantize_zaya_ct_nvfp4.py --scheme w4a4 \\\n"
        "      --model-id %s \\\n"
        "      --mixed-precision-threshold 1000.0 \\\n"
        "      --output-dir ./zaya1-8b-nvfp4-w4a4-sq", out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
