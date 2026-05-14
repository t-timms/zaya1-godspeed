#!/bin/bash
# Quality smoke test: generate text from NVFP4 CT model and verify output quality
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

SCRIPT="/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/scripts/wsl_quality_smoke_test.py"

/home/ttimm/vllm-env/bin/python3 "$SCRIPT" 2>&1
EXIT=$?
echo "Exit code: $EXIT"
exit $EXIT
