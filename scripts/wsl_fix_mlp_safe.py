#!/usr/bin/env python3
"""Fix: add mlp. fallback to all param lookups (no tensor ambiguity)."""

p = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
c = open(p).read()

# Line 925: param = params_dict[param_name] (linear_fc1)
c = c.replace(
    "                    param = params_dict[param_name]\n                    half",
    '                    param = params_dict.get(param_name)\n                    if param is None:\n                        alt = param_name.replace(".zaya_block.", ".mlp.zaya_block.")\n                        param = params_dict.get(alt)\n                    if param is None:\n                        raise KeyError(f"{param_name}")\n                    half',
)

# Line 938: param = params_dict[param_name] (linear_fc2)
c = c.replace(
    "                    param = params_dict[param_name]\n                    fused_moe_module.weight_loader(\n                        param, loaded_weight",
    '                    param = params_dict.get(param_name)\n                    if param is None:\n                        alt = param_name.replace(".zaya_block.", ".mlp.zaya_block.")\n                        param = params_dict.get(alt)\n                    if param is None:\n                        raise KeyError(f"{param_name}")\n                    fused_moe_module.weight_loader(\n                        param, loaded_weight',
)

# Line 955: param = params_dict[chkpt_weight_name] (else)
c = c.replace(
    "            param = params_dict[chkpt_weight_name]",
    '            param = params_dict.get(chkpt_weight_name)\n            if param is None:\n                alt = chkpt_weight_name.replace(".zaya_block.", ".mlp.zaya_block.")\n                param = params_dict.get(alt)\n            if param is None:\n                raise KeyError(f"{chkpt_weight_name}")',
)

open(p, "w").write(c)
print("NO-TENSOR-AMBIGUITY FALLBACKS ADDED")
