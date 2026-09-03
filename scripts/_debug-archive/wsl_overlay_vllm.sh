#!/bin/bash
VLLM=/root/vllm-env/lib/python3.12/site-packages/vllm
SRC=/tmp/zaya-vllm/vllm
source ~/vllm-env/bin/activate

cp -v $SRC/model_executor/layers/mamba/cca.py $VLLM/model_executor/layers/mamba/cca.py
cp -v $SRC/v1/attention/backends/cca_attn.py $VLLM/v1/attention/backends/cca_attn.py
cp -v $SRC/model_executor/models/zaya.py $VLLM/model_executor/models/zaya.py
cp -v $SRC/tool_parsers/zaya_tool_parser.py $VLLM/tool_parsers/zaya_tool_parser.py
cp -v $SRC/transformers_utils/configs/zaya.py $VLLM/transformers_utils/configs/zaya.py

echo "Overlay done"
ls -la $VLLM/model_executor/models/zaya.py
