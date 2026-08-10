"""Paired analysis of the Phase A suite: McNemar on discordant items.

Why not a two-proportion z-test: the two checkpoints are quantizations of ONE
base model evaluated on the SAME items, so their per-item outcomes are strongly
correlated. An unpaired test throws that correlation away and charges variance
for it - which is why ARC-Easy's +1.81 pp landed at p=0.18 on n=2376. McNemar
conditions on the concordant items and tests only where the two disagree, so the
same data yields a far sharper test.

    b = 6.02 GB correct, 9.46 GB wrong
    c = 6.02 GB wrong,   9.46 GB correct

Under H0 (quantization change is inconsequential), each discordant item is a
coin flip, so b ~ Binomial(b+c, 0.5). Exact binomial rather than the chi-square
approximation: discordant counts here may be small, where chi-square is
anti-conservative.

Reports acc and acc_norm separately - they moved in opposite directions on
ARC-Easy, and that disagreement is itself informative.

Usage:
    python3 scripts/analyze_phase_a.py --dir results/phase_a \
                                       --new 6.02GB --control 9.46GB
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any

TASKS = ("hellaswag", "arc_challenge", "winogrande", "piqa")


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial p-value for H0: P(discordance favours either) = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def load_samples(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = rec.get("doc_id")
            if doc_id is not None:
                out[int(doc_id)] = rec
    return out


def correctness(rec: dict, metric: str) -> bool | None:
    """Prefer lm-eval's own grade; fall back to argmax of the loglikelihoods."""
    val = rec.get(metric)
    if isinstance(val, (int, float)):
        return bool(round(float(val)))
    lls = rec.get("lls") or []
    target = rec.get("target")
    if lls and isinstance(target, int) and 0 <= target < len(lls):
        return max(range(len(lls)), key=lambda i: lls[i]) == target
    return None


def analyse(newp: Path, ctlp: Path, metric: str) -> dict[str, Any] | None:
    new, ctl = load_samples(newp), load_samples(ctlp)
    shared = sorted(set(new) & set(ctl))
    if not shared:
        return None

    a = b = c = d = 0  # both right / new only / ctl only / both wrong
    for doc_id in shared:
        n_ok, c_ok = correctness(new[doc_id], metric), correctness(ctl[doc_id], metric)
        if n_ok is None or c_ok is None:
            continue
        if n_ok and c_ok:
            a += 1
        elif n_ok and not c_ok:
            b += 1
        elif not n_ok and c_ok:
            c += 1
        else:
            d += 1

    n = a + b + c + d
    if n == 0:
        return None

    acc_new = (a + b) / n
    acc_ctl = (a + c) / n
    diff = acc_new - acc_ctl
    p = exact_mcnemar_p(b, c)

    # Wald CI on the paired difference, which depends only on the discordant cells.
    var = (b + c - ((b - c) ** 2) / n) / (n**2) if n else 0.0
    se = var**0.5 if var > 0 else 0.0

    return {
        "metric": metric,
        "n_paired": n,
        "both_correct": a,
        "new_only": b,
        "control_only": c,
        "both_wrong": d,
        "discordant": b + c,
        "acc_new": acc_new,
        "acc_control": acc_ctl,
        "diff_pp": diff * 100,
        "ci95_pp": [(diff - 1.96 * se) * 100, (diff + 1.96 * se) * 100],
        "mcnemar_exact_p": p,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/phase_a")
    ap.add_argument("--new", default="6.02GB")
    ap.add_argument("--control", default="9.46GB")
    args = ap.parse_args()

    root = Path(args.dir)
    rows: list[dict] = []

    for task in TASKS:
        newp = root / f"{args.new}__{task}.samples.jsonl"
        ctlp = root / f"{args.control}__{task}.samples.jsonl"
        if not newp.exists() or not ctlp.exists():
            missing = "new" if not newp.exists() else "control"
            print(f"{task:<16} SKIP - {missing} samples missing")
            continue
        for metric in ("acc", "acc_norm"):
            r = analyse(newp, ctlp, metric)
            if r:
                r["task"] = task
                rows.append(r)

    if not rows:
        print("\nNo paired results yet.")
        return 1

    print()
    print("=" * 104)
    print(f"Phase A - paired McNemar:  {args.new} (new) vs {args.control} (control)")
    print("=" * 104)
    hdr = (
        f"{'task':<15} {'metric':<9} {'n':>6} {'new':>8} {'ctl':>8} "
        f"{'diff pp':>8} {'95% CI':>18} {'b':>5} {'c':>5} {'p':>9}"
    )
    print(hdr)
    print("-" * 104)
    for r in rows:
        ci = f"[{r['ci95_pp'][0]:+.2f},{r['ci95_pp'][1]:+.2f}]"
        print(
            f"{r['task']:<15} {r['metric']:<9} {r['n_paired']:>6} "
            f"{r['acc_new'] * 100:>7.2f}% {r['acc_control'] * 100:>7.2f}% "
            f"{r['diff_pp']:>+8.2f} {ci:>18} {r['new_only']:>5} {r['control_only']:>5} "
            f"{r['mcnemar_exact_p']:>9.4f}"
        )
    print("-" * 104)
    print("b = new correct / control wrong.  c = new wrong / control correct.")
    print("p is the two-sided exact McNemar test on the b vs c split.")

    # Bonferroni across the tests actually run - 8 comparisons invites a false
    # positive at alpha=0.05 otherwise.
    k = len(rows)
    alpha = 0.05 / k
    sig = [r for r in rows if r["mcnemar_exact_p"] < alpha]
    print(f"\nBonferroni alpha = 0.05/{k} = {alpha:.4f}")
    if sig:
        print("SIGNIFICANT after correction:")
        for r in sig:
            print(f"  {r['task']} / {r['metric']}: {r['diff_pp']:+.2f} pp (p={r['mcnemar_exact_p']:.4g})")
        print("\n-> A real difference exists. 'No measurable cost' no longer holds as stated.")
    else:
        print("No comparison significant after correction.")
        print("-> Consistent with 'no measurable cost'. Quote the CIs, not just the p-values:")
        print("   they bound how large a regression the data can still hide.")

    out = root / "phase_a_paired_summary.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
