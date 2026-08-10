#!/usr/bin/env bash
# Decode-latency/throughput measurement via vLLM's OWN benchmark CLI.
#
# This replaces an earlier hand-rolled harness that reported 6.5 tok/s for a
# workload `vllm bench latency` measures at ~105 tok/s - wrong by 17x. Use the
# maintained tool; it is one command and anyone can reproduce it.
#
# Notes that matter for the numbers:
#   - --no-enable-prefix-caching is REQUIRED: ZAYA's CCA state is not cacheable
#     and vLLM defaults prefix caching on.
#   - CUDA-graph memory estimation is left ENABLED (vLLM default). The eval
#     scripts disable it via VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 to free
#     VRAM under enforce_eager; doing that here would measure a different engine
#     configuration and is how an earlier run produced a non-comparable 111 tok/s.
#   - Run on a WARM compile cache. A cold cache measures the CUDA compiler:
#     ~198 s first load vs ~58 s warm, on the same checkpoint.
#   - The log filename includes the memory fraction, so a later successful run
#     cannot overwrite the evidence from a failed one.
#
# Usage:
#   scripts/bench_latency.sh <model-path> <label> [batch] [gpu-mem]
#
# Measured 2026-08-09, RTX 5070 Ti 16 GB, vLLM 0.20.2:
#   ./zaya1-8b-nvfp4-w4a4-arcbase  6.02GB  1  0.85  -> 104.7 tok/s, KV 156,981 tok
#   ./zaya1-8b-nvfp4-w4a4          9.46GB  1  0.92  -> 105.3 tok/s, KV  46,802 tok
# The 9.46 GB checkpoint FAILS at 0.85 (no KV memory) and at 1.0 (desktop holds
# ~1.3 GiB), so the two cannot be run at the same fraction on this card.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source "$HOME/vllm-env/bin/activate" || exit 1

MODEL="${1:?usage: bench_latency.sh <model-path> <label> [batch] [gpu-mem]}"
LABEL="${2:?missing label}"
BATCH="${3:-1}"
GPUMEM="${4:-0.85}"

mkdir -p results/throughput
OUT="results/throughput/official_${LABEL}_b${BATCH}_gm${GPUMEM}.json"
LOG="results/throughput/official_${LABEL}_b${BATCH}_gm${GPUMEM}.log"

echo "model=$MODEL label=$LABEL batch=$BATCH gpu_mem=$GPUMEM"
echo "full log -> $LOG"

vllm bench latency \
  --model "$MODEL" \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization "$GPUMEM" \
  --max-model-len 4096 \
  --no-enable-prefix-caching \
  --input-len 128 \
  --output-len 256 \
  --batch-size "$BATCH" \
  --num-iters-warmup 1 \
  --num-iters 3 \
  --output-json "$OUT" > "$LOG" 2>&1
rc=$?

echo "rc=$rc"
grep -iE 'GPU KV cache size|Estimated CUDA graph|Graph capturing finished|Model loading took' "$LOG" | sed 's/^/  /' | tail -6

if [ $rc -ne 0 ]; then
  echo "  FAILED - root cause:"
  grep -iE 'ValueError|RuntimeError|OutOfMemory|CUDA out of memory' "$LOG" \
    | grep -viE 'Engine core initialization failed' | sed 's/^/    /' | head -4
  exit $rc
fi

python3 - "$OUT" "$BATCH" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); batch = int(sys.argv[2]); toks = 256 * batch
avg = d["avg_latency"]; lat = d.get("latencies", [])
print(f"  avg latency : {avg:.3f} s for {toks} tokens")
print(f"  throughput  : {toks/avg:.1f} tok/s")
if lat:
    print(f"  spread      : {100*(max(lat)-min(lat))/avg:.2f}% across {len(lat)} iters")
PY
