#!/usr/bin/env python3
"""Fix ALL params_dict lookups in zaya.py load_weights with mlp. fallback."""

path = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
content = open(path).read()

# Fix 1: line 925 - local_experts linear_fc1 param lookup
old1 = "                    param = params_dict[param_name]\n                    half"
new1 = '                    param = params_dict.get(param_name) or params_dict.get(\n                        param_name.replace(".zaya_block.", ".mlp.zaya_block.")\n                    )\n                    if param is None:\n                        raise KeyError(f"Parameter {param_name} not found")\n                    half'
content = content.replace(old1, new1)

# Fix 2: line 938 - local_experts linear_fc2 param lookup
old2 = "                    param = params_dict[param_name]\n                    fused_moe_module.weight_loader(\n                        param, loaded_weight"
new2 = '                    param = params_dict.get(param_name) or params_dict.get(\n                        param_name.replace(".zaya_block.", ".mlp.zaya_block.")\n                    )\n                    if param is None:\n                        raise KeyError(f"Parameter {param_name} not found")\n                    fused_moe_module.weight_loader(\n                        param, loaded_weight'

# Fix 3: line 955 - else branch (already has fallback, but let's make it consistent)
# Skip if already patched
if "CT naming v3" in content:
    print("ALREADY PATCHED")
else:
    content = content.replace(old1, new1)
    if "mlp.zaya_block" in content:
        print("Fix 1 applied")
    content = content.replace(old2, new2)
    if "mlp.zaya_block" in content and content.count("mlp.zaya_block") >= 2:
        print("Fix 2 applied")
    content = content.replace(
        "CT fallback: vLLM-native names may omit", "CT naming v3 fallback: vLLM-native names may omit"
    )
    open(path, "w").write(content)
    print("ALL FIXES APPLIED")
