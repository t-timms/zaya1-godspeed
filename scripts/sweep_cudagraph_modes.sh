#!/usr/bin/env bash
# Sweep vLLM CUDA-graph modes, gating every throughput number on OUTPUT CORRECTNESS.
#
# Why this exists
# ---------------
# RESEARCH.md 5.14 established that CUDA graph capture produces numerically wrong
# output for this model on SM120, and `--enforce-eager` became mandatory. But that
# sweep varied the MoE BACKEND (flashinfer_cutlass / cutlass / marlin) - it never
# varied the CUDA-graph MODE. vLLM exposes five:
#
#   NONE                nothing captured (this is what --enforce-eager gives you)
#   PIECEWISE           captures everything EXCEPT attention and incompatible ops
#   FULL                full graphs, non-uniform batches
#   FULL_DECODE_ONLY    full graphs, uniform decode batches only
#   FULL_AND_PIECEWISE  vLLM's default: full for decode, piecewise for prefill
#
# The untested hypothesis: the corruption is in ZAYA's CCA state update, not in
# the FP4 MoE kernels. CCA carries convolutional state (config `cca: true`,
# `mamba_cache_dtype: float32`) and in-place state buffers are a known graph-capture
# hazard. That would explain why Marlin - which barely touches the FP4 MoE path -
# also failed. PIECEWISE is precisely the mode that leaves attention eager while
# capturing the rest, so if CCA is the culprit, PIECEWISE should generate correctly
# AND recover most of the graph win.
#
# Also worth re-testing plainly: the corruption was diagnosed 2026-08-14 on
# vLLM 0.20.2. This box now runs vLLM 0.26.0 / FlashInfer 0.6.14 / torch 2.11.0+cu130.
# The default mode may simply have been fixed upstream in the interim.
#
# Why the gate is structural
# --------------------------
# A mode that fails coherence NEVER reaches the benchmark step. That is deliberate:
# the retracted 102.6 tok/s figure was a real measurement of a broken configuration,
# and the only reliable defence is to make it impossible to produce a speed number
# for output nobody verified.
#
# Usage:
#   scripts/sweep_cudagraph_modes.sh [model] [gpu-mem]
#
# Cost: ~1 h GPU. Each mode loads the model (~58 s warm) + gate + 3 bench iters.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source "$HOME/vllm-env/bin/activate" || exit 1

MODEL="${1:-Ttimms/zaya1-8b-nvfp4-w4a4-uniform}"
GPUMEM="${2:-0.85}"
RESULTS="results/cudagraph_sweep"
SUMMARY="$RESULTS/summary.tsv"
mkdir -p "$RESULTS"

echo "model=$MODEL gpu_mem=$GPUMEM"
echo "vllm=$(python -c 'import vllm;print(vllm.__version__)' 2>/dev/null)" \
     "flashinfer=$(python -c 'import flashinfer;print(flashinfer.__version__)' 2>/dev/null)"
echo

# ---------------------------------------------------------------- reference ---
# Captured under NONE (enforce_eager) - the only configuration confirmed correct.
if [ ! -f "$RESULTS/reference.json" ]; then
  echo "=== capturing reference (mode=NONE, enforce_eager) ==="
  python scripts/check_coherence.py --model "$MODEL" --mode NONE \
      --gpu-mem "$GPUMEM" --write-reference || {
    echo "FATAL: reference capture failed or produced garbage. Stop - nothing"
    echo "downstream is trustworthy without a clean reference."
    exit 1
  }
  echo
fi

printf 'mode\tcoherence\tmean_sim\ttok_s_b1\tnote\n' > "$SUMMARY"

for MODE in NONE PIECEWISE FULL_DECODE_ONLY FULL FULL_AND_PIECEWISE; do
  echo "================ mode=$MODE ================"

  # ---- step 1: correctness gate. Must pass before any timing happens. ----
  if [ "$MODE" = "NONE" ]; then
    # NONE is the reference configuration by construction.
    COH="PASS"; SIM="1.0000"
  else
    if python scripts/check_coherence.py --model "$MODEL" --mode "$MODE" \
         --gpu-mem "$GPUMEM"; then
      COH="PASS"
    else
      COH="$(python - "$RESULTS/coherence_$MODE.json" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
print(json.loads(p.read_text())["verdict"] if p.exists() else "ERROR")
PY
)"
    fi
    SIM="$(python - "$RESULTS/coherence_$MODE.json" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
print(json.loads(p.read_text())["mean_similarity"] if p.exists() else "n/a")
PY
)"
  fi

  if [ "$COH" != "PASS" ]; then
    echo "  coherence=$COH -> SKIPPING benchmark. No throughput number will be"
    echo "  recorded for this mode; a fast wrong answer is not a result."
    printf '%s\t%s\t%s\t%s\t%s\n' "$MODE" "$COH" "$SIM" "-" "benchmark skipped" >> "$SUMMARY"
    echo
    continue
  fi

  # ---- step 2: identical methodology to scripts/bench_latency.sh ----
  echo "  coherence=PASS -> benchmarking"
  if [ "$MODE" = "NONE" ]; then
    EXTRA=(--enforce-eager)
  else
    EXTRA=(--compilation-config "{\"cudagraph_mode\": \"$MODE\"}")
  fi

  OUT="$RESULTS/bench_${MODE}_b1_gm${GPUMEM}.json"
  LOG="$RESULTS/bench_${MODE}_b1_gm${GPUMEM}.log"
  vllm bench latency \
    --model "$MODEL" \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization "$GPUMEM" \
    --max-model-len 4096 \
    --no-enable-prefix-caching \
    --input-len 128 \
    --output-len 256 \
    --batch-size 1 \
    --num-iters-warmup 1 \
    --num-iters 3 \
    "${EXTRA[@]}" \
    --output-json "$OUT" > "$LOG" 2>&1
  rc=$?

  if [ $rc -ne 0 ]; then
    echo "  benchmark FAILED rc=$rc"
    grep -iE 'ValueError|RuntimeError|OutOfMemory|CUDA out of memory|IllegalAddress' "$LOG" \
      | sed 's/^/    /' | head -4
    printf '%s\t%s\t%s\t%s\t%s\n' "$MODE" "$COH" "$SIM" "-" "bench rc=$rc" >> "$SUMMARY"
    echo
    continue
  fi

  TOKS="$(python - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"{256 / d['avg_latency']:.1f}")
PY
)"
  echo "  throughput: $TOKS tok/s"
  printf '%s\t%s\t%s\t%s\t%s\n' "$MODE" "$COH" "$SIM" "$TOKS" "ok" >> "$SUMMARY"
  echo
done

echo "================ summary ================"
column -t -s $'\t' "$SUMMARY"
echo
echo "Written to $SUMMARY"
echo
echo "Before quoting ANY number above: a PASS is a similarity gate, not a proof."
echo "Read results/cudagraph_sweep/coherence_<mode>.json for the actual generated"
echo "text of the winning mode, and re-run a real eval (HellaSwag or HumanEval)"
echo "under that mode before it goes near a model card. RESEARCH.md 5.14 is what"
echo "happens when a throughput number ships ahead of that step."
