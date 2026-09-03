#!/bin/bash
# Run apply_professional_fixes.py inside WSL where vLLM is installed.
# Must be run AFTER vLLM is installed in /home/ttimm/vllm-env
set -e
cd /mnt/c/Users/ttimm/Documents/Project\ Portfolio/zaya1-godspeed
export PATH=/usr/local/cuda/bin:$PATH
source /home/ttimm/vllm-env/bin/activate
python scripts/apply_professional_fixes.py "$@"
