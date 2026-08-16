#!/usr/bin/env bash
# Runs the three budget-forced generative benchmarks unattended.
#
# These complement the loglikelihood suite (hellaswag/arc/winogrande/piqa),
# which scores pre-written continuations and therefore cannot detect whether
# the model can actually *produce* correct maths or code - the blind spot that
# let a checkpoint incapable of forming a sentence score 61% on HellaSwag.
#
# Design follows AGENTS.md's benchmarking protocol:
#   - one process per benchmark, so one failure cannot take down the others
#   - success judged by the artifact on disk parsing as JSON, never by exit
#     code (vLLM can abort at teardown *after* writing a valid result) and
#     never by file size (a 200-byte floor once rejected a valid 188-byte run)
#   - re-runs resume: a stage with a valid artifact is skipped, not repeated
#   - the complete log always goes to a file; only the console view is filtered
#
# Order is cheapest-first so a partial run still leaves usable results.
#
# Usage (WSL, from the repo root):
#     bash scripts/run_budget_forced_suite.sh
#     MODEL=./zaya1-8b-nvfp4-w4a4 bash scripts/run_budget_forced_suite.sh
#     THINK_BUDGET=8192 bash scripts/run_budget_forced_suite.sh
#     FORCE=1 bash scripts/run_budget_forced_suite.sh    # ignore existing artifacts
set -uo pipefail

MODEL="${MODEL:-./zaya1-8b-nvfp4-w4a4-arcbase}"
THINK_BUDGET="${THINK_BUDGET:-4096}"
MMLU_N="${MMLU_N:-700}"          # stratified subset; -1 for the full 12,032
PY="${PY:-$HOME/vllm-env/bin/python3}"
FORCE="${FORCE:-0}"

# The context must hold the prompt + the full reasoning budget + the answer.
# Left at the scripts' 8192 default, a THINK_BUDGET of 8192 would be cut off by
# the context limit rather than the budget — silently measuring the wrong thing
# while every log line still looks correct. Derive it instead, with headroom.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(( THINK_BUDGET * 2 ))}"
if [ "$MAX_MODEL_LEN" -lt 8192 ]; then MAX_MODEL_LEN=8192; fi
# SMOKE_N caps every stage to N items and redirects output to a separate
# directory. Use it to verify the orchestration end-to-end (~10 min) before
# committing to a multi-hour run — the plumbing is what fails at 3am, not the
# maths. Smoke artifacts never collide with real ones.
SMOKE_N="${SMOKE_N:-}"

if [ -n "$SMOKE_N" ]; then
  RESULTS_DIR="${RESULTS_DIR:-results/budget_forced_smoke}"
  MMLU_N="$SMOKE_N"
else
  RESULTS_DIR="${RESULTS_DIR:-results/budget_forced}"
fi
LOG_DIR="${LOG_DIR:-$RESULTS_DIR/logs}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

TAG="$(basename "$MODEL")"
START_TS="$(date +%Y%m%d-%H%M%S)"
SUMMARY="$RESULTS_DIR/SUMMARY-${TAG}-${START_TS}.txt"

echo "suite start   : $(date -Is)"        | tee "$SUMMARY"
echo "model         : $MODEL"             | tee -a "$SUMMARY"
echo "think budget  : $THINK_BUDGET"      | tee -a "$SUMMARY"
echo "max model len : $MAX_MODEL_LEN"     | tee -a "$SUMMARY"
echo "mmlu subset n : $MMLU_N"            | tee -a "$SUMMARY"
if [ -n "$SMOKE_N" ]; then
  echo "MODE          : SMOKE (n=$SMOKE_N per stage) — results are NOT publishable" | tee -a "$SUMMARY"
fi
echo "" | tee -a "$SUMMARY"

# Environment fingerprint - a stack change with unchanged numbers still
# invalidates prior claims, so record it alongside the results.
{
  echo "--- environment ---"
  "$PY" -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null
  "$PY" -c "import torch; print('torch', torch.__version__, torch.version.cuda)" 2>/dev/null
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null
  ( cd "$HOME/vllm-src" 2>/dev/null && echo "vllm-src commit $(git rev-parse --short HEAD)" )
  echo ""
} | tee -a "$SUMMARY"

# A stage is "already done" only if its artifact exists AND parses as JSON.
artifact_ok() {
  local path="$1"
  [ -f "$path" ] || return 1
  "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" "$path" >/dev/null 2>&1
}

run_stage() {
  local name="$1" script="$2" artifact="$3"
  shift 3

  if [ "$FORCE" != "1" ] && artifact_ok "$artifact"; then
    echo "[$name] SKIP - valid artifact already at $artifact" | tee -a "$SUMMARY"
    return 0
  fi

  local log="$LOG_DIR/${name}-${TAG}-${START_TS}.log"
  echo "[$name] start $(date -Is)  -> $log" | tee -a "$SUMMARY"

  local t0 t1
  t0=$(date +%s)
  # Full output to the log file. Never pipe a fallible subprocess through a
  # filter - the filtered line is always the one naming the cause.
  "$PY" "$script" --model "$MODEL" --think-budget "$THINK_BUDGET" \
    --max-model-len "$MAX_MODEL_LEN" --output "$artifact" "$@" >"$log" 2>&1
  local rc=$?
  t1=$(date +%s)

  if artifact_ok "$artifact"; then
    echo "[$name] OK   $(( (t1 - t0) / 60 ))m  (exit $rc, artifact valid)" | tee -a "$SUMMARY"
    "$PY" - "$artifact" <<'PYEOF' | tee -a "$SUMMARY"
import json, sys

d = json.load(open(sys.argv[1]))
n = d.get("n", 0)

# Lead with the headline metric *and its interval* on one line. A point
# estimate alone is not interpretable: 15/20 reads as 75% but is really
# [53.1, 88.8], and a 75-vs-80 pair from this suite tested at p=1.0000.
for metric, ci_key in (("accuracy", "accuracy_ci95"), ("pass_at_1", "pass_at_1_ci95")):
    if metric in d:
        line = f"        {metric}={d[metric] * 100:.1f}%"
        ci = d.get(ci_key)
        if ci:
            line += f"  95% CI [{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]"
        line += f"  n={n}"
        print(line)

diag = ("self_closed_think", "hit_think_budget_ceiling", "recovered_by_trace_fallback")
present = [f"{k}={d[k]}" for k in diag if k in d]
if present:
    print("        " + "  ".join(present))
    ceiling = d.get("hit_think_budget_ceiling")
    if ceiling is not None and n and ceiling / n > 0.9:
        print(f"        NOTE: {ceiling}/{n} hit the think-budget ceiling — consider a larger --think-budget")
PYEOF
  else
    echo "[$name] FAIL $(( (t1 - t0) / 60 ))m  (exit $rc, no valid artifact) - see $log" | tee -a "$SUMMARY"
    tail -n 15 "$log" | sed 's/^/        | /' | tee -a "$SUMMARY"
  fi
  echo "" | tee -a "$SUMMARY"
}

# In smoke mode every stage is capped; otherwise HumanEval and GSM8K run their
# full sets and only MMLU-Pro takes a subset (it is 12,032 items).
SMOKE_ARG=()
[ -n "$SMOKE_N" ] && SMOKE_ARG=(--n "$SMOKE_N")

# Cheapest first: a partial run still leaves the most results on disk.
run_stage humaneval scripts/eval_humaneval_budget_forced.py \
  "$RESULTS_DIR/humaneval-${TAG}.json" "${SMOKE_ARG[@]}"

run_stage mmlu_pro scripts/eval_mmlu_pro_budget_forced.py \
  "$RESULTS_DIR/mmlu_pro-${TAG}.json" --n "$MMLU_N"

run_stage gsm8k scripts/eval_gsm8k_budget_forced.py \
  "$RESULTS_DIR/gsm8k-${TAG}.json" "${SMOKE_ARG[@]}"

echo "suite end     : $(date -Is)" | tee -a "$SUMMARY"
echo "summary       : $SUMMARY"    | tee -a "$SUMMARY"
