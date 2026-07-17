#!/usr/bin/env bash
# Overnight eval runner for ZAYA1-8B W4A4 (budget-forced, reasoning-aware).
#
# Runs two GPU jobs SEQUENTIALLY (single 16 GB GPU — they cannot overlap):
#   1. IFEval, full 541 prompts, budget-forced + think-stripped   (~75-90 min)
#   2. GPQA-Diamond, n=100, think_budget=12000 @ 16k ctx          (~3-3.5 hr)
# Total ~4.5-5 hr. IFEval runs first (shorter, higher value-per-hour); a failure
# there surfaces early in the logs instead of after the long GPQA leg.
#
# Launch (from Windows or WSL):
#   wsl bash "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/scripts/run_overnight_evals.sh"
#
# Everything is logged; safe to disconnect. Results land in results/ (see SUMMARY at end).

set -u  # undefined-var guard; do NOT set -e — one leg failing must not skip the other.

PROJ="/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed"
VENV="/home/ttimm/vllm-env/bin/activate"
STAMP="$(date +%Y%m%d_%H%M%S)"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

# shellcheck disable=SC1090
source "$VENV"
cd "$PROJ" || { echo "FATAL: cannot cd to $PROJ"; exit 1; }
mkdir -p results

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Overnight evals starting (run id $STAMP) ==="
log "GPU free at start:"
nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader

# ---------------------------------------------------------------------------
log ">>> LEG 1/2: IFEval (full 541, budget-forced)"
IFEVAL_LOG="results/overnight_${STAMP}_ifeval.log"
python3 scripts/eval_ifeval_budget_forced.py \
    --think-budget 2048 --response-max 2048 --max-model-len 8192 \
    --output "results/overnight_${STAMP}_ifeval.json" \
    > "$IFEVAL_LOG" 2>&1
log "IFEval leg exit=$? -> $IFEVAL_LOG"

# ---------------------------------------------------------------------------
log ">>> LEG 2/2: GPQA-Diamond (n=100, think_budget=12000)"
GPQA_LOG="results/overnight_${STAMP}_gpqa.log"
python3 scripts/eval_gpqa_budget_forced.py \
    --n 100 --think-budget 12000 --max-model-len 16384 \
    > "$GPQA_LOG" 2>&1
log "GPQA leg exit=$? -> $GPQA_LOG"

# ---------------------------------------------------------------------------
echo
echo "########################  SUMMARY (run $STAMP)  ########################"
echo "--- IFEval ---"
grep -aE "prompt_level_strict_acc|prompt_level_loose_acc|inst_level|self-closed|n=" "$IFEVAL_LOG" 2>/dev/null | tail -8
echo "--- GPQA-Diamond (n=100) ---"
grep -aE "accuracy:|self-closed|reference" "$GPQA_LOG" 2>/dev/null | tail -5
echo "########################################################################"
log "=== Overnight evals complete ==="
