"""Fix uncalibrated input_global_scale entries in an NVFP4 W4A4 checkpoint.

Modules that were never activated during calibration (e.g., MoE experts not
routed to by the calibration data) have input_global_scale = 0.0 (CT default).
When those experts are activated at inference, the block_scale = igs * vec_max / 6
= 0 * anything = 0, causing zero / NaN outputs and catastrophically wrong logits.

Fix: for each bad module, compute a fallback scale from the median of calibrated
modules in the same layer with the same Linear type (linear_fc1 or linear_fc2).
If no calibrated same-layer peers exist, expand the search radius to ±N layers.

Also fixes the 7 modules with massively negative/large garbage values (float32
denormals or overflow from uninitialized memory).

Usage:
    python3 scripts/fix_uncalibrated_igs.py [--checkpoint PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import shutil
import statistics
from pathlib import Path

import safetensors.torch as st
import torch

CHECKPOINT_DEFAULT = "zaya1-8b-nvfp4-w4a4-sq-mrgptq"
IGS_VALID_MIN = 0.5  # below this → treated as bad (igs<0.5 implies max_act>5376)
IGS_VALID_MAX = 1400.0  # above this → treated as bad


def _is_bad(val: float) -> bool:
    return val < IGS_VALID_MIN or val > IGS_VALID_MAX or val != val  # NaN check


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    safetensors_path = ckpt / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(safetensors_path)

    print(f"Loading {safetensors_path} ...")
    tensors = st.load_file(str(safetensors_path))

    layer_re = re.compile(r"model\.layers\.(\d+)\..*\.(linear_fc[12])\.input_global_scale")

    # ── Separate good from bad ──────────────────────────────────────────────
    good_by_key: dict[tuple[int, str], list[float]] = {}  # (layer, fc_type) → values
    bad_keys: list[str] = []

    for key in tensors:
        if not key.endswith(".input_global_scale"):
            continue
        val = tensors[key].item()
        m = layer_re.match(key)
        if m:
            layer_idx = int(m.group(1))
            fc_type = m.group(2)
            if _is_bad(val):
                bad_keys.append(key)
            else:
                good_by_key.setdefault((layer_idx, fc_type), []).append(val)

    total_igs = sum(1 for k in tensors if k.endswith(".input_global_scale"))
    print(f"Total IGS keys: {total_igs}  |  bad: {len(bad_keys)}  |  good: {total_igs - len(bad_keys)}")

    if not bad_keys:
        print("Nothing to fix.")
        return

    # ── Compute fallback per bad key ────────────────────────────────────────
    patched: dict[str, torch.Tensor] = {}
    fallback_missing: list[str] = []

    for key in bad_keys:
        m = layer_re.match(key)
        if not m:
            # Key doesn't match expected pattern — use global median as last resort
            all_good = [v for vs in good_by_key.values() for v in vs]
            fallback = statistics.median(all_good) if all_good else 5.0
            patched[key] = torch.tensor(fallback, dtype=torch.float32)
            continue

        layer_idx = int(m.group(1))
        fc_type = m.group(2)

        # Search same layer first, then expand radius
        fallback = None
        for radius in range(0, 40):
            candidates: list[float] = []
            for delta in range(0, radius + 1) if radius == 0 else [-radius, radius]:
                neighbor = layer_idx + delta
                candidates.extend(good_by_key.get((neighbor, fc_type), []))
            if candidates:
                fallback = statistics.median(candidates)
                break

        if fallback is None:
            # Absolute fallback: global median across both fc types
            all_good = [v for vs in good_by_key.values() for v in vs]
            fallback = statistics.median(all_good) if all_good else 5.0
            fallback_missing.append(key)

        patched[key] = torch.tensor(fallback, dtype=torch.float32)

    if fallback_missing:
        print(f"WARNING: {len(fallback_missing)} keys needed global fallback (no same-type neighbors found)")
        for k in fallback_missing[:5]:
            print(f"  {k}")

    # ── Report ─────────────────────────────────────────────────────────────
    print("\nPatched values sample:")
    for key in list(patched.keys())[:8]:
        old_val = tensors[key].item()
        new_val = patched[key].item()
        print(f"  {key.split('model.layers.')[1]}")
        print(f"    {old_val:.3e}  →  {new_val:.3f}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # ── Backup + write ──────────────────────────────────────────────────────
    backup = safetensors_path.with_suffix(".safetensors.bak.uncalib")
    if not backup.exists():
        print(f"\nBacking up to {backup.name} ...")
        shutil.copy2(safetensors_path, backup)

    for key, tensor in patched.items():
        tensors[key] = tensor

    print(f"Saving patched checkpoint to {safetensors_path} ...")
    st.save_file(tensors, str(safetensors_path))
    print(f"Done. Patched {len(patched)} IGS entries.")


if __name__ == "__main__":
    main()
