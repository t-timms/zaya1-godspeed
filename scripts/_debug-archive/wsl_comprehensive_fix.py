#!/usr/bin/env python3
"""Comprehensive fix: add mlp. fallback to ALL weight lookups in load_weights.

CT-quantized safetensors use: model.layers.N.zaya_block.experts.*
vLLM module hierarchy uses:  model.layers.N.mlp.zaya_block.experts.*

This adds f-string-based alt_prefix/alt_name fallbacks at all 4 lookup points:
1. fused_moe_modules dict lookup
2. params_dict for linear_fc1
3. params_dict for linear_fc2
4. params_dict for else branch
"""

p = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
c = open(p).read()

# Fix 1: fused_moe_modules lookup — try both with and without mlp prefix
c = c.replace(
    "                fused_moe_module = fused_moe_modules.get(fused_moe_prefix)\n                if fused_moe_module is None:",
    '                fused_moe_module = fused_moe_modules.get(fused_moe_prefix)\n                if fused_moe_module is None:\n                    alt_moe = fused_moe_prefix.replace(".zaya_block.", ".mlp.zaya_block.")\n                    fused_moe_module = fused_moe_modules.get(alt_moe)\n                if fused_moe_module is None:',
)

# Fix 2: linear_fc1 param_name — try mlp-prefixed alternative
c = c.replace(
    '                    param_name = f"{fused_moe_prefix}.w13_weight"\n                    param = params_dict[param_name]',
    '                    param_name = f"{fused_moe_prefix}.w13_weight"\n                    param = params_dict.get(param_name)\n                    if param is None:\n                        alt = param_name.replace(".zaya_block.", ".mlp.zaya_block.")\n                        param = params_dict.get(alt)\n                    if param is None:\n                        raise KeyError(f"{param_name}")',
)

# Fix 3: linear_fc2 param_name — try mlp-prefixed alternative
c = c.replace(
    '                    param_name = f"{fused_moe_prefix}.w2_weight"\n                    param = params_dict[param_name]',
    '                    param_name = f"{fused_moe_prefix}.w2_weight"\n                    param = params_dict.get(param_name)\n                    if param is None:\n                        alt = param_name.replace(".zaya_block.", ".mlp.zaya_block.")\n                        param = params_dict.get(alt)\n                    if param is None:\n                        raise KeyError(f"{param_name}")',
)

# Fix 4: else branch chkpt_weight_name — try mlp-prefixed alternative
c = c.replace(
    "            param = params_dict[chkpt_weight_name]",
    '            param = params_dict.get(chkpt_weight_name)\n            if param is None:\n                alt = chkpt_weight_name.replace(".zaya_block.", ".mlp.zaya_block.")\n                param = params_dict.get(alt)\n            if param is None:\n                raise KeyError(f"{chkpt_weight_name}")',
)

open(p, "w").write(c)
print("COMPREHENSIVE MLP PREFIX FALLBACKS APPLIED")
