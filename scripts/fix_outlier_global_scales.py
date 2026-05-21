"""Patch input_global_scale for outlier layers to prevent FP8 scale saturation.

Root cause
----------
The current convention stores ``input_global_scale = 2688 / max_abs(x)``.
The CUTLASS NVFP4 kernel computes per-block activation scales as::

    per_block_scale = max_abs_block / input_global_scale    (clamped to FP8 range)

FP8_E4M3 max = 448, so saturation occurs when::

    max_abs / input_global_scale > 448
    ⇒  max_abs^2 / 2688 > 448
    ⇒  max_abs > sqrt(2688 * 448) ≈ 1097

When the per-block FP8 scale saturates at 448, the effective quantization
step becomes 448 / input_global_scale. For the worst outlier in our
checkpoint (max_abs=8896 at L75 expert 1 linear_fc2), the current
input_global_scale=0.302 produces a step size of 448/0.302 ≈ 1483.
With the corrected scale of 19.86, the step is 448/19.86 ≈ 23 — a **64×
improvement** in quantization resolution for that block.

Fix
---
For every layer with ``max_abs > 1097``, replace::

    input_global_scale = 2688 / max_abs

with::

    input_global_scale = max(2688 / max_abs, max_abs / 448)

This guarantees ``per_block_scale ≤ 448``, eliminating saturation while
preserving the original dequantization identity for non-saturating blocks.

No changes to ``weight_packed``, ``weight_scale_fp8``, or
``weight_global_scale`` are needed — this is purely an activation-side fix.

Usage
-----
    python3 scripts/fix_outlier_global_scales.py [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import safetensors.torch as st
import torch

# Constants
FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
# Saturation threshold = sqrt(FP8_E4M3_MAX * FP8_E4M3_MAX * FP4_E2M1_MAX)
# Actually: threshold = sqrt(2688 * 448) = sqrt(FP8_MAX * FP4_MAX * 448)
# Where 2688 = FP8_E4M3_MAX * FP4_E2M1_MAX = 448 * 6
GLOBAL_SCALE_NUM = FP8_E4M3_MAX * FP4_E2M1_MAX  # = 2688.0
SATURATION_THRESHOLD = (GLOBAL_SCALE_NUM * FP8_E4M3_MAX) ** 0.5  # ≈ 1097.0

DEFAULT_CKPT = Path(
    "/mnt/c/Users/ttimm/Documents/Project Portfolio/"
    "zaya1-godspeed/zaya1-8b-nvfp4-w4a4/model.safetensors"
)
DEFAULT_CONFIG = DEFAULT_CKPT.with_name("config.json")


def compute_corrected_igs(max_abs: float) -> float:
    """Return the outlier-safe input_global_scale for a given max_abs.

    The corrected formula guarantees no per-block FP8 scale exceeds
    FP8_E4M3_MAX (448), eliminating saturation-induced precision loss.
    """
    old_igs = GLOBAL_SCALE_NUM / max_abs
    safe_igs = max_abs / FP8_E4M3_MAX
    return max(old_igs, safe_igs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fix input_global_scale for outlier layers."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CKPT,
        help="Path to model.safetensors"
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Path to config.json"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print changes without modifying the checkpoint"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Apply changes without creating a backup"
    )
    args = parser.parse_args()

    # ── Load checkpoint metadata ──────────────────────────────────────
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found at {args.checkpoint}", file=sys.stderr)
        return 1

    config_path = args.config
    if not config_path.exists():
        print(f"WARNING: config not found at {config_path}", file=sys.stderr)
        # Non-fatal; we just won't update config metadata

    print(f"Loading checkpoint: {args.checkpoint}")
    state = st.load_file(str(args.checkpoint), device="cpu")

    # ── Identify all input_global_scale entries ───────────────────────
    igs_keys = sorted(
        k for k in state if k.endswith("input_global_scale") and state[k].numel() == 1
    )
    print(f"Found {len(igs_keys)} per-Linear input_global_scale entries")

    # ── Analyze and patch ─────────────────────────────────────────────
    patches: list[tuple[str, float, float, float, float]] = []
    # (key, max_abs, old_igs, new_igs, loss_factor)

    total_saturation_loss = 0.0
    total_fix_gain = 0.0

    for key in igs_keys:
        old_igs = state[key].float().item()
        if old_igs <= 0:
            continue
        max_abs = GLOBAL_SCALE_NUM / old_igs
        new_igs = compute_corrected_igs(max_abs)

        # Loss = how much the per-block scale exceeds FP8 max (linear)
        # 1.0 means no saturation; >1 means saturation
        current_per_block_max = max_abs / old_igs
        saturation_ratio = current_per_block_max / FP8_E4M3_MAX
        # Fix gain = ratio by which we improve the quantization step
        fix_gain_ratio = new_igs / old_igs if new_igs > old_igs else 1.0

        if new_igs > old_igs * 1.01:  # >1% change
            patches.append((key, max_abs, old_igs, new_igs, saturation_ratio))
            total_saturation_loss += saturation_ratio - 1.0
            total_fix_gain += fix_gain_ratio - 1.0

    # Sort by severity (largest saturation ratio first)
    patches.sort(key=lambda x: x[4], reverse=True)

    # ── Report ────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"Outlier Analysis: {len(patches)} layers need input_global_scale correction")
    print(f"Saturation threshold (max_abs > {SATURATION_THRESHOLD:.0f}): "
          f"per-block FP8 scale overflows FP8_E4M3_MAX={FP8_E4M3_MAX}")
    print(f"{'=' * 80}\n")

    print(f"{'#':>3}  {'max_abs(x)':>10}  {'old_igs':>10}  {'new_igs':>10}  "
          f"{'sat_ratio':>10}  {'fix_gain':>10}  module")
    print(f"{'-' * 3}  {'-' * 10}  {'-' * 10}  {'-' * 10}  "
          f"{'-' * 10}  {'-' * 10}  {'-' * 50}")

    for i, (key, ma, old, new, sat_ratio) in enumerate(patches):
        fix_gain = new / old
        short = key.replace("model.layers.", "L").replace(".input_global_scale", "")
        print(f"{i + 1:>3}  {ma:>10.1f}  {old:>10.4f}  {new:>10.4f}  "
              f"{sat_ratio:>10.2f}  {fix_gain:>10.2f}  {short}")

    print("\nSummary:")
    print(f"  Layers patched:     {len(patches)}")
    print(f"  Total sat loss:     {total_saturation_loss:.2f}x")
    print(f"  Total fix gain:     {total_fix_gain:.2f}x")
    print(f"  Worst offender:     max_abs={patches[0][1]:.0f} at "
          f"{patches[0][0].replace('.input_global_scale','')}")
    print(f"  Worst fix gain:     {patches[0][3]/patches[0][2]:.1f}x "
          f"better quantization resolution")

    # ── Apply ─────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[Dry-run mode] No changes written.")
        return 0

    if not patches:
        print("\nNo layers need correction. Nothing to do.")
        return 0

    # Create backup
    backup_path = args.checkpoint.with_suffix(".safetensors.bak.outlier")
    if not args.force:
        if backup_path.exists():
            print(f"\nERROR: Backup already exists at {backup_path}. "
                  f"Use --force to overwrite.", file=sys.stderr)
            return 1

    print(f"\nCreating backup: {backup_path}")
    shutil.copy2(str(args.checkpoint), str(backup_path))

    # Apply patches
    print(f"Patching {len(patches)} input_global_scale entries...")
    for key, ma, old, new, sat_ratio in patches:
        state[key] = torch.tensor(new, dtype=state[key].dtype)
        print(f"  {key}: {old:.4f} → {new:.4f}")

    # Save
    print(f"\nSaving patched checkpoint to {args.checkpoint}")
    st.save_file(state, str(args.checkpoint), metadata={"format": "pt"})
    print(f"Done. Backup at {backup_path}")

    # ── Optionally patch config.json to document the fix ──────────────
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            quant_cfg = config.get("quantization_config", {})
            quant_cfg["outlier_mitigation"] = (
                f"patched {len(patches)} input_global_scale entries "
                f"with formula max(2688/max_abs, max_abs/448) to prevent "
                f"FP8 scale saturation"
            )
            if not args.dry_run:
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                print("Updated config.json with outlier_mitigation note")
        except Exception as e:
            print(f"WARNING: could not update config.json: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
