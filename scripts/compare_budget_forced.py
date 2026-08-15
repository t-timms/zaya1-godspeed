"""Paired McNemar comparison between two budget-forced eval runs.

Two checkpoints evaluated on the same items is a PAIRED design. Comparing only
the aggregate accuracies discards the pairing and forces an unpaired test —
the mistake that produced an unusable ARC-Easy result (+1.81 pp, p=0.18) and
wasted the run. The budget-forced scripts preserve per-item outcomes exactly
so this test is possible.

`~/scripts/bench-mcnemar.py` does this for lm-eval `samples_*.jsonl` output;
it cannot read these results, which carry their own per-item schema, hence
this companion rather than a format shim.

Usage:
    python3 scripts/compare_budget_forced.py \\
        results/budget_forced/gsm8k-zaya1-8b-nvfp4-w4a4-arcbase.json \\
        results/budget_forced/gsm8k-zaya1-8b-nvfp4-w4a4.json \\
        --label-a uniform --label-b mixed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# GSM8K/MMLU-Pro store "per_question"; HumanEval stores "per_task".
ITEM_KEYS = ("per_question", "per_task")
# Stable per-item identity. GSM8K has no id field, so fall back to position —
# valid only because both runs iterate the same dataset in the same order with
# the same seed, which the scripts guarantee.
ID_KEYS = ("task_id", "key", "doc_id")
OUTCOME_KEYS = ("correct", "passed")


def load_outcomes(path: Path) -> tuple[dict[str, int], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = None
    for k in ITEM_KEYS:
        if k in data:
            items = data[k]
            break
    if items is None:
        raise SystemExit(f"{path}: no per-item outcomes ({'/'.join(ITEM_KEYS)}) — cannot pair")

    outcomes: dict[str, int] = {}
    for pos, item in enumerate(items):
        ident = next((str(item[k]) for k in ID_KEYS if k in item), str(pos))
        val = next((item[k] for k in OUTCOME_KEYS if k in item), None)
        if val is None:
            raise SystemExit(f"{path}: item {ident} has no outcome field ({'/'.join(OUTCOME_KEYS)})")
        outcomes[ident] = int(bool(val))
    return outcomes, data


def mcnemar(a: dict[str, int], b: dict[str, int], label_a: str, label_b: str) -> None:
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("no overlapping items between the two runs")
    only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
    if only_a or only_b:
        print(
            f"warning: {only_a} items only in {label_a}, {only_b} only in {label_b} — comparing the {len(shared)} shared"
        )

    n = len(shared)
    acc_a = sum(a[i] for i in shared) / n
    acc_b = sum(b[i] for i in shared) / n
    n10 = sum(1 for i in shared if a[i] == 1 and b[i] == 0)  # a right, b wrong
    n01 = sum(1 for i in shared if a[i] == 0 and b[i] == 1)  # b right, a wrong
    delta = (acc_b - acc_a) * 100

    print(f"\nn = {n} paired items")
    print(f"  {label_a:<12} {acc_a * 100:.2f}%")
    print(f"  {label_b:<12} {acc_b * 100:.2f}%")
    print(f"  delta        {delta:+.2f} pp  (positive favours {label_b})")
    print(f"  discordant   {label_a}-only-correct={n10}  {label_b}-only-correct={n01}")

    disc = n10 + n01
    if disc == 0:
        print("  identical on every item — no discordant pairs, no test to run")
        return

    try:
        from scipy.stats import binomtest
    except ImportError:
        print("  scipy not available — install it for the exact-binomial p-value and CI")
        return

    res = binomtest(n01, disc, 0.5)
    ci = res.proportion_ci(confidence_level=0.95)
    # Convert the proportion CI on discordant pairs back to a percentage-point
    # difference in accuracy: delta = (2p - 1) * disc / n.
    lo = (2 * ci.low - 1) * disc / n * 100
    hi = (2 * ci.high - 1) * disc / n * 100
    verdict = "SIGNIFICANT" if res.pvalue < 0.05 else "not significant"
    print(f"  95% CI       [{lo:+.2f}, {hi:+.2f}] pp")
    print(f"  p-value      {res.pvalue:.4f}  ({verdict} at alpha=0.05)")
    if res.pvalue >= 0.05:
        print("  → cannot distinguish these checkpoints on this task at this sample size.")
        print("    Report the CI, not the point estimate — a non-significant delta is not a finding.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    a, data_a = load_outcomes(args.run_a)
    b, data_b = load_outcomes(args.run_b)

    print(f"paired McNemar: {args.label_a} vs {args.label_b}")
    print(f"  {args.label_a}: {data_a.get('model', args.run_a)}")
    print(f"  {args.label_b}: {data_b.get('model', args.run_b)}")

    # Budget mismatch makes the comparison meaningless — different amounts of
    # allowed reasoning is a different experiment, not a different checkpoint.
    ta, tb = data_a.get("think_budget"), data_b.get("think_budget")
    if ta != tb:
        print(f"\n  ⚠ think_budget differs ({ta} vs {tb}) — these runs are NOT comparable.")
        print("    Re-run both at the same budget before drawing any conclusion.")

    mcnemar(a, b, args.label_a, args.label_b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
