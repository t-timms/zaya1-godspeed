#!/usr/bin/env bash
# Starts vLLM serving this project's NVFP4 W4A4 checkpoint.
#
# This is the exact configuration validated in RESEARCH.md 5.14-5.18: the
# correctness fix and the speed win, both required to reproduce the
# project's numbers rather than a plausible-looking but wrong or slow setup.
set -euo pipefail

MODEL="${MODEL:-Ttimms/zaya1-8b-nvfp4-w4a4}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.90}"

# --enforce-eager is MANDATORY, not a performance knob. CUDA graph capture
# computes numerically WRONG output for this model on SM120 - a served model
# without it returns fluent-looking garbage under sampling. See README.md
# "Known Issue: CUDA graph capture corrupts output on SM120" / RESEARCH.md
# 5.14 before ever removing this flag.
#
# --speculative-config (n-gram): a free, lossless 2.2x speedup on coding-edit
# workloads (read a file, echo most of it back with a small change) -
# validated in RESEARCH.md 5.18 via scripts/bench_ngram_coding_edit.py.
# Rejection sampling makes this exact-output-preserving, unlike
# --enforce-eager: there is no quality tradeoff for enabling it. It gives ~0
# gain on free-form generation (no prompt/output token overlap to exploit),
# so if your workload is mostly open-ended writing rather than editing, this
# flag will not hurt but won't help much either.
vllm serve "$MODEL" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enforce-eager \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 5, "prompt_lookup_min": 2}'
