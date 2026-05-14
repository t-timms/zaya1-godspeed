#!/bin/bash
# Overlay Zyphra vLLM fork Python files onto stock vLLM 0.20.2
set -e
VLLM=/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm
SRC=/tmp/zaya-vllm/vllm

echo "=== Zyphra vLLM Overlay ==="

mkdir -p "$VLLM"/model_executor/layers/mamba
mkdir -p "$VLLM"/v1/attention/backends
mkdir -p "$VLLM"/model_executor/models
mkdir -p "$VLLM"/tool_parsers
mkdir -p "$VLLM"/transformers_utils/configs

cp -v "$SRC"/model_executor/layers/mamba/cca.py          "$VLLM"/model_executor/layers/mamba/cca.py
cp -v "$SRC"/v1/attention/backends/cca_attn.py            "$VLLM"/v1/attention/backends/cca_attn.py
cp -v "$SRC"/model_executor/models/zaya.py                "$VLLM"/model_executor/models/zaya.py
cp -v "$SRC"/tool_parsers/zaya_tool_parser.py             "$VLLM"/tool_parsers/zaya_tool_parser.py
cp -v "$SRC"/transformers_utils/configs/zaya.py           "$VLLM"/transformers_utils/configs/zaya.py

echo ""
echo "Overlay done. Files:"
ls -la "$VLLM"/model_executor/models/zaya.py
ls -la "$VLLM"/model_executor/layers/mamba/cca.py
ls -la "$VLLM"/v1/attention/backends/cca_attn.py
ls -la "$VLLM"/tool_parsers/zaya_tool_parser.py
ls -la "$VLLM"/transformers_utils/configs/zaya.py
