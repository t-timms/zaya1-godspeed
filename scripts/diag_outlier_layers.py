r"""Diagnostic: analyze input_global_scale distribution in the W4A4 checkpoint.

Identifies outlier layers where max_act >> median, causing precision loss
in per-block activation quantization.

Run from WSL:
    python3 /mnt/c/Users/ttimm/Documents/Project\ Portfolio/zaya1-godspeed/scripts/diag_outlier_layers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import safetensors.torch as st

FP8_MAX = 448.0
FP4_MAX = 6.0
GS_NUM = FP8_MAX * FP4_MAX  # 2688

CKPT = Path(
    "/mnt/c/Users/ttimm/Documents/Project Portfolio/"
    "zaya1-godspeed/zaya1-8b-nvfp4-w4a4/model.safetensors"
)


def main() -> int:
    print(f"Loading {CKPT} ...")
    state = st.load_file(str(CKPT), device="cpu")

    igs_keys = sorted(k for k in state if k.endswith("input_global_scale"))
    print(f"Found {len(igs_keys)} input_global_scale entries\n")

    # Analyze all entries
    rows: list[tuple[float, float, float, str]] = []
    # (max_act, igs, p50_block_scale, module_name)

    for k in igs_keys:
        igs = state[k].float().item()
        if igs <= 0:
            continue
        max_act = GS_NUM / igs
        # Per-block FP8 scale for a block at 50% of max_act
        p50_scale = igs * (max_act * 0.5) / 6.0
        # Per-block FP8 scale for a block at 10% of max_act (typical)
        p10_scale = igs * (max_act * 0.1) / 6.0
        rows.append((max_act, igs, p50_scale, p10_scale, k))

    rows.sort(key=lambda x: -x[0])

    # Compute stats
    max_act_vals = [r[0] for r in rows]
    median_ma = sorted(max_act_vals)[len(max_act_vals) // 2]
    mean_ma = sum(max_act_vals) / len(max_act_vals)

    igs_vals = [r[1] for r in rows]
    median_igs = sorted(igs_vals)[len(igs_vals) // 2]

    print(f"max_act stats:  median={median_ma:.1f}  mean={mean_ma:.1f}  "
          f"max={max_act_vals[0]:.1f}  min={max_act_vals[-1]:.1f}")
    print(f"igs stats:      median={median_igs:.6f}  "
          f"min={igs_vals[-1]:.6f}  max={igs_vals[0]:.6f}")
    print()

    # Find layers where max_act > 3 * median (outlier threshold)
    outlier_threshold = median_ma * 3
    outliers = [r for r in rows if r[0] > outlier_threshold]
    print(f"Outliers (max_act > {outlier_threshold:.0f} = 3x median): "
          f"{len(outliers)} layers\n")

    print(f"{'max_act':>10}  {'igs':>12}  {'p50_fp8_scale':>14}  "
          f"{'p10_fp8_scale':>14}  module")
    print("-" * 90)
    for ma, igs, p50, p10, k in outliers:
        short = k.replace("model.layers.", "L").replace(".input_global_scale", "")
        print(f"{ma:>10.1f}  {igs:>12.6f}  {p50:>14.2f}  {p10:>14.2f}  {short}")

    print(f"\nTotal layers analyzed: {len(rows)}")
    print(f"Outlier layers: {len(outliers)} ({100*len(outliers)/len(rows):.1f}%)")
    print("Recommendation:")
    print("  Current IGS convention is correct. Outlier layers have low IGS")
    print("  because max_act is high. This prevents FP8 scale saturation but")
    print("  reduces precision for moderate-activation blocks.")
    print("  A per-channel IGS scheme would mitigate this at the cost of")
    print("  custom kernel changes.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
