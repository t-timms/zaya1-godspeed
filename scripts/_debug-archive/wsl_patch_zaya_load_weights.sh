#!/bin/bash
# Patch vLLM zaya.py load_weights to handle compressed_tensors parameter names
ZAYA=/root/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py

python3 << 'PYEOF'
import re

with open('/root/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py') as f:
    content = f.read()

# Fix 1: w13_weight → try w13_weight_packed first, fallback to w13_weight
old = """                if parts[-2] == "linear_fc1":
                    param_name = f"{fused_moe_prefix}.w13_weight"
                    param = params_dict[param_name]"""
new = """                if parts[-2] == "linear_fc1":
                    param_name = f"{fused_moe_prefix}.w13_weight"
                    # compressed_tensors: try _packed variant first
                    if param_name not in params_dict and f"{param_name}_packed" in params_dict:
                        param_name = f"{param_name}_packed"
                    param = params_dict[param_name]"""
content = content.replace(old, new)

# Fix 2: w2_weight → try w2_weight_packed first
old = """                elif parts[-2] == "linear_fc2":
                    param_name = f"{fused_moe_prefix}.w2_weight"
                    param = params_dict[param_name]"""
new = """                elif parts[-2] == "linear_fc2":
                    param_name = f"{fused_moe_prefix}.w2_weight"
                    if param_name not in params_dict and f"{param_name}_packed" in params_dict:
                        param_name = f"{param_name}_packed"
                    param = params_dict[param_name]"""
content = content.replace(old, new)

# Fix 3: Also handle weight_scale and weight_global_scale for MoE
# These are loaded separately by the weight_loader
# No change needed - weight_loader handles the scale parameters

with open('/root/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py', 'w') as f:
    f.write(content)

print("Patched zaya.py for compressed_tensors MoE naming")
PYEOF
