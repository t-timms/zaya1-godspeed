#!/bin/bash
# Smoke test: load NVFP4 CT ZAYA1-8B model via vLLM
export PATH=/usr/local/cuda/bin:/usr/bin:/usr/local/bin:/sbin:/bin
export CUDA_VISIBLE_DEVICES=0

MODEL_DIR="/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct"
SCRIPT="/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/scripts/wsl_smoke_test.py"

/home/ttimm/vllm-env/bin/python3 "$SCRIPT" 2>&1
EXIT=$?
echo "Exit code: $EXIT"
exit $EXIT
