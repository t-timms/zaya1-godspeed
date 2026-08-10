#!/usr/bin/env bash
# Phase A status. A file, not an inline command: PowerShell parses inline python
# passed through `wsl bash -c` as PowerShell and chokes on `for`/`$`/quotes.
set -uo pipefail
cd "$HOME/zaya1-godspeed" || exit 1
source "$HOME/vllm-env/bin/activate" 2>/dev/null

python3 - <<'PY'
import json, glob, os

rows = []
for f in sorted(glob.glob("results/phase_a/*__*.json")):
    if f.endswith("phase_a_paired_summary.json"):
        continue
    d = json.load(open(f))
    m = d.get("metrics", {})
    rows.append((
        d.get("label"), d.get("task"), d.get("n_samples"),
        m.get("acc,none"), m.get("acc_stderr,none"),
        m.get("acc_norm,none"), m.get("acc_norm_stderr,none"),
        d.get("elapsed_sec", 0) / 60.0,
    ))

if not rows:
    print("no completed tasks yet")
else:
    print(f"{'label':<9} {'task':<15} {'n':>6} {'acc':>8} {'+/-':>7} {'acc_norm':>9} {'+/-':>7} {'min':>6}")
    print("-" * 74)
    for lbl, task, n, a, ase, an, anse, mins in rows:
        f2 = lambda v: f"{v*100:.2f}" if isinstance(v, (int, float)) else "-"
        print(f"{lbl:<9} {task:<15} {n:>6} {f2(a):>8} {f2(ase):>7} {f2(an):>9} {f2(anse):>7} {mins:>6.1f}")

print()
for lbl in ("6.02GB", "9.46GB"):
    # Written out rather than as a comprehension: the two-`if` form evaluated
    # getsize(p) before the walrus that binds p, since comprehension conditions
    # run left to right.
    done = 0
    for t in ("hellaswag", "arc_challenge", "winogrande", "piqa"):
        p = f"results/phase_a/{lbl}__{t}.samples.jsonl"
        if os.path.exists(p) and os.path.getsize(p) > 0:
            done += 1
    print(f"  {lbl}: {done}/4 tasks with per-doc records")
PY

echo
echo "--- driver progress ---"
grep -E '===|###|-->|OK -|FAILED|SKIP ' results/phase_a/driver.log | tail -6

echo
echo "--- current activity ---"
tail -1 results/phase_a/driver.log | cut -c1-140
pgrep -af 'run_phase_a' | grep -v pgrep | head -1 | cut -c1-120 || echo "  (no eval process running)"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  GPU: /'

echo
echo "--- control checkpoint ---"
if [ -d "./zaya1-8b-nvfp4-w4a4" ]; then
  echo "  present"
else
  echo "  ABSENT - needs: hf auth login, then bash fetch_control.sh"
fi
