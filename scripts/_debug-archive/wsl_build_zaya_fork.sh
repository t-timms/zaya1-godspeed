#!/bin/bash
# SOTA Professional Path: Install Zyphra vLLM fork from local clone
# Already have: torch 2.11.0 + CUDA 13.2 deps from stock vLLM install
set -e

export PATH=/usr/local/cuda/bin:/usr/bin:/usr/local/bin:/sbin:/bin
source /home/ttimm/vllm-env/bin/activate

echo "=== Building Zyphra vLLM fork ==="
echo "CUDA: $(which nvcc)"
echo "Python: $(which python3)"

cd /tmp/zaya-vllm

# Clean any previous build artifacts
rm -rf build/ dist/ *.egg-info vllm.egg-info 2>/dev/null || true

# Verify the branch
echo "Branch: $(git branch --show-current)"
echo "HEAD: $(git log --oneline -1)"

# Install (should be fast since deps are already satisfied)
pip install -e . 2>&1 | tail -30
echo ""
echo "=== Build complete ==="
pip show vllm | grep -E 'Name|Version|Location'
