#!/usr/bin/env python3
"""FULL FIX: zaya.py load_weights for CT-quantized Zaya model.

Fixes:
1. Router layers bypass CT quantization (prefix-based skip in compressed_tensors.py)
2. Router mlp layers get proper prefixes in zaya.py
3. FusedMoE module lookup includes mlp. fallback
4. CT-quantized MoE uses _packed parameter suffix
5. All param lookups include mlp. fallback
"""

import sys

# ─── Fix 1 & 2: Router prefixes in zaya.py ───
zp = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
zc = open(zp).read()

# Router mlp prefixes (already applied via wsl_fix_router_prefix.py pattern)
if 'prefix=f"{prefix}.router_mlp.0"' not in zc:
    print("ERROR: Router mlp prefixes missing. Run wsl_fix_router_prefix.py first")
    sys.exit(1)

# ─── Fix 3 & 4 & 5: FusedMoE lookups + packed suffix + mlp fallback ───

# 3a: fused_moe_prefix lookup — add mlp. alt
zc = zc.replace(
    "                fused_moe_module = fused_moe_modules.get(fused_moe_prefix)\n                if fused_moe_module is None:",
    '                fused_moe_module = fused_moe_modules.get(fused_moe_prefix)\n                if fused_moe_module is None:\n                    alt_moe = fused_moe_prefix.replace(".zaya_block.", ".mlp.zaya_block.")\n                    fused_moe_module = fused_moe_modules.get(alt_moe)\n                if fused_moe_module is None:',
)

# 4a: linear_fc1 — try w13_weight, then w13_weight_packed, then mlp alt
zc = zc.replace(
    '                    param_name = f"{fused_moe_prefix}.w13_weight"\n                    param = params_dict[param_name]',
    '                    param_name = f"{fused_moe_prefix}.w13_weight"\n                    param = params_dict.get(param_name)\n                    if param is None:\n                        param_name = f"{fused_moe_prefix}.w13_weight_packed"\n                        param = params_dict.get(param_name)\n                    if param is None:\n                        param_name = param_name.replace(".zaya_block.", ".mlp.zaya_block.")\n                        param = params_dict.get(param_name)\n                    if param is None:\n                        raise KeyError(f"FusedMoE w13 param not found for {fused_moe_prefix}")',
)

# 4b: linear_fc2 — try w2_weight, then w2_weight_packed, then mlp alt
zc = zc.replace(
    '                    param_name = f"{fused_moe_prefix}.w2_weight"\n                    param = params_dict[param_name]',
    '                    param_name = f"{fused_moe_prefix}.w2_weight"\n                    param = params_dict.get(param_name)\n                    if param is None:\n                        param_name = f"{fused_moe_prefix}.w2_weight_packed"\n                        param = params_dict.get(param_name)\n                    if param is None:\n                        param_name = param_name.replace(".zaya_block.", ".mlp.zaya_block.")\n                        param = params_dict.get(param_name)\n                    if param is None:\n                        raise KeyError(f"FusedMoE w2 param not found for {fused_moe_prefix}")',
)

# 5: else branch — try mlp alt
zc = zc.replace(
    "            param = params_dict[chkpt_weight_name]",
    '            param = params_dict.get(chkpt_weight_name)\n            if param is None:\n                alt = chkpt_weight_name.replace(".zaya_block.", ".mlp.zaya_block.")\n                param = params_dict.get(alt)\n            if param is None:\n                raise KeyError(f"{chkpt_weight_name}")',
)

open(zp, "w").write(zc)
print("ALL LOAD_WEIGHTS FIXES APPLIED")
