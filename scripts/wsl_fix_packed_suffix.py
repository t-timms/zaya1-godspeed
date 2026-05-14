#!/usr/bin/env python3
"""Fix: CT-quantized FusedMoE uses _packed suffix for weight parameters."""

p = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
c = open(p).read()

# Fix linear_fc1: try "w13_weight" first, then "w13_weight_packed"
c = c.replace(
    '                    param_name = f"{fused_moe_prefix}.w13_weight"',
    '                    param_name = f"{fused_moe_prefix}.w13_weight"\n                    if param_name not in params_dict:\n                        param_name = f"{fused_moe_prefix}.w13_weight_packed"\n                    if param_name not in params_dict:\n                        alt = param_name.replace(".zaya_block.", ".mlp.zaya_block.")\n                        if alt in params_dict:\n                            param_name = alt',
)

# Fix linear_fc2: try "w2_weight" first, then "w2_weight_packed"
c = c.replace(
    '                    param_name = f"{fused_moe_prefix}.w2_weight"',
    '                    param_name = f"{fused_moe_prefix}.w2_weight"\n                    if param_name not in params_dict:\n                        param_name = f"{fused_moe_prefix}.w2_weight_packed"\n                    if param_name not in params_dict:\n                        alt = param_name.replace(".zaya_block.", ".mlp.zaya_block.")\n                        if alt in params_dict:\n                            param_name = alt',
)

open(p, "w").write(c)
print("ADDED _packed suffix fallback for CT MoE weights")
