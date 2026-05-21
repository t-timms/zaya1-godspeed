"""Expert-Balanced Sample Selection (EBSS) for NVFP4 W4A4 calibration.

MoEQuant (arXiv:2505.03804) identifies that standard calibration data
under-represents infrequently-activated MoE experts: with top-1 routing,
16 experts, and 977 calibration samples × 40 MoE layers, each expert is
activated on average only ~61 times per layer. Experts with fewer activations
receive less accurate input_global_scale calibration.

EBSS resamples the calibration data so that all 16 experts receive
approximately equal activation counts. This is done by:
  1. Loading only the router weights from the W4A4 checkpoint (BF16, small).
  2. Running a lightweight router-forward-only pass over all calibration samples.
  3. Building a per-sample, per-layer expert-activation profile.
  4. Greedily selecting/reordering samples to maximize minimum expert coverage.
  5. Saving a rebalanced calibration_data_ebss.pt with the same shape.

Usage (WSL, vllm-env active):
    python3 scripts/build_calibration_ebss.py \\
        --input  data/calibration/arcmix/calibration_data.pt \\
        --output data/calibration/arcmix_ebss/calibration_data.pt \\
        --checkpoint zaya1-8b-nvfp4-w4a4

    # Dry-run: show coverage stats without writing output
    python3 scripts/build_calibration_ebss.py --input ... --dry-run
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

DEFAULT_CHECKPOINT = (
    "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-w4a4"
)
DEFAULT_INPUT = (
    "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/"
    "data/calibration/arcmix/calibration_data.pt"
)
DEFAULT_OUTPUT = (
    "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/"
    "data/calibration/arcmix_ebss/calibration_data.pt"
)

NUM_EXPERTS = 16  # ZAYA1-8B: 16 experts per MoE layer, top-1 routing


# ─────────────────────────────────────────────────────────────────
# Router weight extraction
# ─────────────────────────────────────────────────────────────────

def load_router_weights(checkpoint_dir: str) -> dict[int, dict[str, torch.Tensor]]:
    """Extract router MLP weights from the checkpoint (BF16, CPU).

    Returns dict: layer_idx → {weight_name: tensor} for each MoE layer.
    Only loads router-related tensors — far smaller than the full checkpoint.
    """
    from safetensors.torch import load_file

    ckpt_path = Path(checkpoint_dir)
    index_path = ckpt_path / "model.safetensors.index.json"

    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        # Collect unique shard files that contain router weights
        router_shards: set[str] = set()
        router_keys: list[str] = []
        for key, shard in weight_map.items():
            if "router" in key and "zaya_block" in key:
                router_shards.add(shard)
                router_keys.append(key)
        logger.info("Found %d router weight keys across %d shards", len(router_keys), len(router_shards))
    else:
        # Single-shard checkpoint
        router_shards = {"model.safetensors"}
        router_keys = None

    # Load only the router shards
    router_tensors: dict[str, torch.Tensor] = {}
    for shard in router_shards:
        shard_path = ckpt_path / shard
        tensors = load_file(str(shard_path))
        for k, v in tensors.items():
            if "router" in k and "zaya_block" in k:
                router_tensors[k] = v.float()  # work in fp32 for stability

    # Group by layer index
    layer_routers: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in router_tensors.items():
        # Key pattern: model.layers.{N}.zaya_block.router.{subname}
        parts = key.split(".")
        try:
            layer_idx = int(parts[2])
        except (IndexError, ValueError):
            continue
        subkey = ".".join(parts[5:])  # strip "model.layers.N.zaya_block.router." prefix
        if layer_idx not in layer_routers:
            layer_routers[layer_idx] = {}
        layer_routers[layer_idx][subkey] = tensor

    logger.info("Loaded router weights for %d MoE layers", len(layer_routers))
    return layer_routers


# ─────────────────────────────────────────────────────────────────
# Lightweight router forward pass
# ─────────────────────────────────────────────────────────────────

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (x.float() * rms).to(weight.dtype) * weight


def router_forward(
    hidden: torch.Tensor,  # [seq_len, hidden_dim] float32
    router_weights: dict[str, torch.Tensor],
    num_experts: int = NUM_EXPERTS,
) -> torch.Tensor:
    """Minimal router forward: hidden → expert_indices [seq_len].

    Matches ZayaRouter: down_proj → rmsnorm_eda → router_mlp (3-layer GELU MLP)
    → softmax → argmax.

    Keys expected in router_weights (after stripping "model.layers.N.zaya_block.router." prefix):
      down_proj.weight, down_proj.bias,
      rmsnorm_eda.weight,
      router_mlp.0.weight, router_mlp.0.bias,
      router_mlp.2.weight, router_mlp.2.bias,
      router_mlp.4.weight
    """
    import torch.nn.functional as F

    # Step 1: down_proj
    down_w = router_weights.get("down_proj.weight")
    if down_w is None:
        raise ValueError(f"Missing down_proj.weight. Keys: {sorted(router_weights.keys())}")
    down_b = router_weights.get("down_proj.bias")
    h = F.linear(hidden, down_w, down_b)  # [S, mlp_expansion]

    # Step 2: rmsnorm_eda
    rn_w = router_weights.get("rmsnorm_eda.weight")
    if rn_w is not None:
        h = _rms_norm(h, rn_w)

    # Step 3: router_mlp (3-layer MLP: Linear-GELU-Linear-GELU-Linear)
    mlp0_w = router_weights.get("router_mlp.0.weight")
    mlp0_b = router_weights.get("router_mlp.0.bias")
    mlp2_w = router_weights.get("router_mlp.2.weight")
    mlp2_b = router_weights.get("router_mlp.2.bias")
    mlp4_w = router_weights.get("router_mlp.4.weight")

    if mlp0_w is None or mlp4_w is None:
        raise ValueError(f"Missing router_mlp weights. Keys: {sorted(router_weights.keys())}")

    h = F.gelu(F.linear(h, mlp0_w, mlp0_b))
    if mlp2_w is not None:
        h = F.gelu(F.linear(h, mlp2_w, mlp2_b))
    logits = F.linear(h, mlp4_w)  # [S, num_experts(+1 for MOD skip)]

    # Clamp to num_experts (MOD adds one extra "skip" slot)
    logits = logits[..., :num_experts]
    expert_indices = logits.argmax(dim=-1)  # [seq_len]
    return expert_indices


# ─────────────────────────────────────────────────────────────────
# Expert activation profiling
# ─────────────────────────────────────────────────────────────────

def profile_expert_activations(
    calibration_tensor: torch.Tensor,  # [N, seq_len]
    layer_routers: dict[int, dict[str, torch.Tensor]],
    embed_weight: torch.Tensor,         # [vocab, hidden]
    num_experts: int = NUM_EXPERTS,
) -> torch.Tensor:
    """Run router-only forward pass over all calibration samples.

    Returns expert_counts: [N_samples, N_moe_layers, num_experts] int32.
    Each entry is the number of tokens in that sample routed to that expert
    in that layer.
    """
    n_samples, seq_len = calibration_tensor.shape
    moe_layer_ids = sorted(layer_routers.keys())
    n_moe_layers = len(moe_layer_ids)

    expert_counts = torch.zeros(n_samples, n_moe_layers, num_experts, dtype=torch.int32)

    logger.info(
        "Profiling expert activations: %d samples × %d MoE layers × %d experts...",
        n_samples, n_moe_layers, num_experts,
    )
    t0 = time.time()

    for sample_idx in range(n_samples):
        if sample_idx % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (n_samples - sample_idx) / max(sample_idx, 1)
            logger.info("  sample %d/%d (%.0fs elapsed, ETA %.0fs)", sample_idx, n_samples, elapsed, eta)

        tokens = calibration_tensor[sample_idx]  # [seq_len]
        # Embed tokens: [seq_len, hidden]
        hidden = embed_weight[tokens].float()  # simple lookup, no position bias

        for layer_pos, layer_idx in enumerate(moe_layer_ids):
            router_weights = layer_routers[layer_idx]
            try:
                expert_idx = router_forward(hidden, router_weights, num_experts)  # [seq_len]
                # Count activations per expert
                for e in range(num_experts):
                    expert_counts[sample_idx, layer_pos, e] = int((expert_idx == e).sum().item())
            except Exception as exc:
                logger.warning("Router forward failed for layer %d sample %d: %s", layer_idx, sample_idx, exc)

    logger.info("Profiling complete in %.0fs", time.time() - t0)
    return expert_counts


# ─────────────────────────────────────────────────────────────────
# Greedy EBSS resampling
# ─────────────────────────────────────────────────────────────────

def ebss_resample(
    expert_counts: torch.Tensor,  # [N, n_moe_layers, num_experts]
    target_n: int,
    num_experts: int = NUM_EXPERTS,
    seed: int = 42,
) -> list[int]:
    """Greedy expert-balanced sample selection.

    Selects `target_n` sample indices (with repetition allowed if needed)
    that maximize the minimum per-expert activation count.

    Algorithm:
      1. Compute current expert coverage: [n_moe_layers, num_experts] sum.
      2. Greedily select the next sample that most improves the least-covered expert.
      3. Repeat until target_n samples selected.

    Returns list of selected sample indices (length target_n).
    """
    torch.manual_seed(seed)
    n_samples = expert_counts.shape[0]

    # Normalize: for each sample, compute its "coverage value" per expert
    # = sum over layers of (tokens routed to that expert in this sample)
    # Shape: [N, num_experts]
    sample_expert_totals = expert_counts.sum(dim=1).float()  # [N, num_experts]

    selected: list[int] = []
    cumulative_coverage = torch.zeros(num_experts, dtype=torch.float32)

    for step in range(target_n):
        # Score each sample: benefit = sum of (1 / (1 + coverage[e])) × sample_expert_totals[s, e]
        # Higher score for samples that activate under-represented experts
        inv_coverage = 1.0 / (1.0 + cumulative_coverage)  # [num_experts]
        scores = (sample_expert_totals * inv_coverage.unsqueeze(0)).sum(dim=1)  # [N]

        # Pick best (add small noise to break ties deterministically)
        noise = torch.rand(n_samples) * 1e-6
        best_idx = int((scores + noise).argmax().item())
        selected.append(best_idx)

        # Update cumulative coverage
        cumulative_coverage += sample_expert_totals[best_idx]

        if step % 100 == 0 and step > 0:
            min_cov = cumulative_coverage.min().item()
            max_cov = cumulative_coverage.max().item()
            logger.info(
                "  EBSS step %d/%d: min_expert_coverage=%.0f max=%.0f ratio=%.2f",
                step, target_n, min_cov, max_cov,
                min_cov / max(max_cov, 1),
            )

    return selected


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Expert-Balanced Sample Selection (EBSS) for NVFP4 calibration"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input calibration_data.pt")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output calibration_data.pt")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="W4A4 checkpoint directory")
    parser.add_argument("--num-experts", type=int, default=NUM_EXPERTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Profile expert activations and print coverage stats, but don't write output.",
    )
    args = parser.parse_args()

    # ── Load calibration data ──────────────────────────────────
    logger.info("Loading calibration data from %s", args.input)
    cal_tensor = torch.load(args.input, map_location="cpu")
    n_samples, seq_len = cal_tensor.shape
    logger.info("Calibration tensor: %d samples × %d tokens", n_samples, seq_len)

    # ── Load router weights + embed_tokens ────────────────────
    logger.info("Loading router weights from %s", args.checkpoint)
    layer_routers = load_router_weights(args.checkpoint)

    from safetensors.torch import load_file as _load_file
    ckpt_path = Path(args.checkpoint)

    # Load embed_tokens.weight
    index_path = ckpt_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        embed_shard = index["weight_map"].get("model.embed_tokens.weight", "model.safetensors")
    else:
        embed_shard = "model.safetensors"

    logger.info("Loading embed_tokens from %s", embed_shard)
    shard_tensors = _load_file(str(ckpt_path / embed_shard))
    embed_weight = shard_tensors["model.embed_tokens.weight"].float()
    logger.info("embed_tokens.weight shape: %s", list(embed_weight.shape))

    # ── Profile expert activations ─────────────────────────────
    expert_counts = profile_expert_activations(
        cal_tensor, layer_routers, embed_weight, args.num_experts
    )

    # ── Coverage statistics ────────────────────────────────────
    # expert_coverage[e] = total tokens routed to expert e across all samples + layers
    coverage = expert_counts.sum(dim=(0, 1)).float()  # [num_experts]
    logger.info("Expert coverage (before EBSS):")
    for e in range(args.num_experts):
        logger.info("  Expert %2d: %7.0f tokens", e, coverage[e].item())
    logger.info(
        "  Min: %.0f  Max: %.0f  Imbalance ratio: %.2f",
        coverage.min().item(),
        coverage.max().item(),
        coverage.min().item() / max(coverage.max().item(), 1),
    )

    if args.dry_run:
        logger.info("Dry-run complete — not writing output.")
        return 0

    # ── EBSS resampling ────────────────────────────────────────
    logger.info("Running greedy EBSS resampling (target=%d samples)...", n_samples)
    selected_indices = ebss_resample(expert_counts, target_n=n_samples, num_experts=args.num_experts, seed=args.seed)

    resampled = cal_tensor[torch.tensor(selected_indices)]

    # Verify coverage after resampling
    resampled_counts = expert_counts[torch.tensor(selected_indices)]
    coverage_after = resampled_counts.sum(dim=(0, 1)).float()
    logger.info("Expert coverage (after EBSS):")
    for e in range(args.num_experts):
        logger.info("  Expert %2d: %7.0f tokens", e, coverage_after[e].item())
    imbalance_before = (coverage.min() / coverage.max()).item()
    imbalance_after = (coverage_after.min() / coverage_after.max()).item()
    logger.info(
        "  Imbalance ratio: %.2f → %.2f (higher is better, 1.0 = perfect balance)",
        imbalance_before, imbalance_after,
    )

    # ── Save output ────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(resampled, out_path)
    logger.info("Saved EBSS-resampled calibration to %s  shape=%s", out_path, list(resampled.shape))

    manifest = {
        "source": args.input,
        "method": "EBSS (MoEQuant arXiv:2505.03804)",
        "n_samples": n_samples,
        "seq_len": seq_len,
        "num_experts": args.num_experts,
        "imbalance_before": round(imbalance_before, 4),
        "imbalance_after": round(imbalance_after, 4),
        "seed": args.seed,
    }
    manifest_path = out_path.parent / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", manifest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
