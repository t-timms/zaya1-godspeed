#!/usr/bin/env bash
# Phase A overnight driver.
#
# Runs each (checkpoint, task) pair as its own process so one failure costs a
# single task, not the night. run_phase_a.py skips any pair whose BOTH artifacts
# already exist, so re-running this script resumes rather than restarting.
#
# Success is judged by artifacts on disk, NOT by exit code: vLLM reliably aborts
# during interpreter teardown ("terminate called without an active exception")
# after results are flushed, so a non-zero rc here does not mean the task failed.
#
# Only one GPU, so this is strictly sequential. The 6.02 GB checkpoint is local
# and runs first; the 9.46 GB control runs only if it has been fetched by then -
# a download finishing during the first half is picked up automatically.

set -uo pipefail

REPO="$HOME/zaya1-godspeed"
cd "$REPO" || exit 1
source "$HOME/vllm-env/bin/activate" || exit 1

RESULTS="$REPO/results/phase_a"
LOG="$RESULTS/driver.log"
mkdir -p "$RESULTS"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

NEW="./zaya1-8b-nvfp4-w4a4-arcbase"   # 6.02 GB, 0 BF16 exemptions
CTL="./zaya1-8b-nvfp4-w4a4"           # 9.46 GB control, pulled from HF
TASKS="hellaswag arc_challenge winogrande piqa"

log "=== Phase A start ==="
log "GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"

run_one () {
  local path="$1" label="$2" task="$3"
  local agg="$RESULTS/${label}__${task}.json"
  local smp="$RESULTS/${label}__${task}.samples.jsonl"

  if [ -s "$agg" ] && [ -s "$smp" ]; then
    log "    already done - skipping"
    return 0
  fi

  python3 scripts/run_phase_a.py --model "$path" --label "$label" --task "$task" >>"$LOG" 2>&1
  local rc=$?

  if [ -s "$agg" ] && [ -s "$smp" ]; then
    log "    OK - $(wc -l < "$smp") per-doc records (rc=$rc, ignored)"
  else
    log "    FAILED - no artifacts written (rc=$rc)"
  fi
  sleep 10   # let the driver reclaim VRAM fully before the next load
}

run_model () {
  local path="$1" label="$2"
  if [ ! -d "$path" ]; then
    log "SKIP $label - checkpoint absent at $path"
    return 0
  fi
  log "### $label ($path)"
  for t in $TASKS; do
    log "--> $label / $t"
    run_one "$path" "$label" "$t"
  done
}

run_model "$NEW" "6.02GB"
run_model "$CTL" "9.46GB"

log "=== evaluation complete ==="
ls -1 "$RESULTS"/*.json 2>/dev/null | sed 's/^/  /' | tee -a "$LOG"

# Only meaningful once both halves exist; prints per-task SKIP lines otherwise.
log "=== paired analysis ==="
python3 scripts/analyze_phase_a.py --new 6.02GB --control 9.46GB 2>&1 | tee -a "$LOG"

log "=== Phase A done ==="
