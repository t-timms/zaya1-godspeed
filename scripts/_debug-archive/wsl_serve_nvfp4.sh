#!/bin/bash
# Serve the NVFP4 Compressed-Tensors ZAYA1-8B model via vLLM.
# Requires: vLLM installed with professional fixes applied.
set -e
MODEL_DIR="/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct-gs16"
export PATH=/usr/local/cuda/bin:$PATH
source /home/ttimm/vllm-env/bin/activate

echo "=== Serving NVFP4 ZAYA1-8B via vLLM ==="
echo "Model: $MODEL_DIR"
echo "Port: 8010"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --port 8010 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --trust-remote-code \
    --enforce-eager \
    --max-num-seqs 1 \
    --tokenizer Zyphra/ZAYA1-8B
