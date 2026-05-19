"""Locate the activation-magnitude outliers in zaya1-8b-nvfp4-w4a4.

Reads each Linear's stored ``input_global_scale`` (= 2688/max_abs(x) per our
convention) and inverts to recover ``max_abs(x)``. Top-K offenders identify
candidates for SmoothQuant-style migration, Hadamard rotation, or selective
BF16 (adding to the ignore list at quantize time).

Run from WSL:
    python3 scripts/analyze_w4a4_outliers.py [--top 30] [--threshold 1000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import safetensors.torch as st

GLOBAL_SCALE_NUM = 2688.0  # = FP8_E4M3_MAX * FP4_E2M1_MAX
DEFAULT_PATH = Path(
    "/mnt/c/Users/ttimm/Documents/Project Portfolio/"
    "zaya1-godspeed/zaya1-8b-nvfp4-w4a4/model.safetensors"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=1000.0)
    args = parser.parse_args()

    state = st.load_file(str(args.path), device="cpu")
    rows: list[tuple[float, str]] = []
    for k, v in state.items():
        if not k.endswith("input_global_scale") or v.numel() != 1:
            continue
        igs = max(v.float().item(), 1e-12)
        max_abs_x = GLOBAL_SCALE_NUM / igs
        rows.append((max_abs_x, k))

    rows.sort(reverse=True)
    total = len(rows)
    above = [r for r in rows if r[0] > args.threshold]

    print(f"# W4A4 activation-outlier ranking ({total} Linears with input_global_scale)")
    print(f"# {len(above)} above max_abs={args.threshold}")
    print()
    print(f"{'rank':>4}  {'max_abs(x)':>11}  {'effective_loss':>14}  module")
    print(f"{'-'*4}  {'-'*11}  {'-'*14}  {'-'*60}")

    for i, (ma, k) in enumerate(rows[: args.top]):
        # If max_abs/6 > 448 the per-block fp8 scale saturates and the
        # outlier block loses dynamic range by max_abs / (6 * 448) =
        # max_abs / 2688.
        loss = ma / 2688.0
        loss_str = f"{loss:>5.2f}x  " if loss > 1.0 else f"  {loss:>5.2f}x"
        short = k.replace("model.layers.", "L").replace(".input_global_scale", "")
        print(f"{i + 1:>4}  {ma:>11.2f}  {loss_str:>14}  {short}")

    print()
    print("Layer-wise top-1 outlier:")
    by_layer: dict[int, tuple[float, str]] = {}
    for ma, k in rows:
        try:
            layer = int(k.split("layers.")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        if layer not in by_layer or ma > by_layer[layer][0]:
            by_layer[layer] = (ma, k)
    for layer in sorted(by_layer.keys()):
        ma, k = by_layer[layer]
        if ma > args.threshold:
            short = k.replace("model.layers.", "L").replace(".input_global_scale", "")
            print(f"  L{layer:>2}: max={ma:>9.2f}  {short}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
