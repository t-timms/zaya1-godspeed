"""Fix scale routing in zaya.py weight loading."""

path = "/home/ttimm/vllm-src/vllm/model_executor/models/zaya.py"
with open(path) as f:
    content = f.read()

# Fix: Route weight_scale keys to correct FusedMoE params
# The checkpoint has w13_weight_scale and w2_weight_scale, but
# zaya.py tries to load them through w13_weight/w2_weight params.

# Fix for linear_fc1 (w13) scale routing
old1 = (
    '                if parts[-2] == "linear_fc1":\n'
    '                    param_name = f"{fused_moe_prefix}.w13_weight"\n'
    "                    if param_name not in params_dict:\n"
    '                        param_name_packed = f"{param_name}_packed"\n'
    "                        if param_name_packed in params_dict:\n"
    "                            param_name = param_name_packed\n"
    "                    param = params_dict[param_name]"
)
new1 = (
    '                if parts[-2] == "linear_fc1":\n'
    "                    # Route weight_scale keys to scale param\n"
    '                    if "weight_scale" in chkpt_weight_name:\n'
    '                        scale_param_name = f"{fused_moe_prefix}.w13_weight_scale"\n'
    "                        if scale_param_name not in params_dict:\n"
    '                            scale_param_name_packed = f"{scale_param_name}_packed"\n'
    "                            if scale_param_name_packed in params_dict:\n"
    "                                scale_param_name = scale_param_name_packed\n"
    "                        scale_param = params_dict[scale_param_name]\n"
    "                        fused_moe_module.weight_loader(\n"
    '                            scale_param, loaded_weight, chkpt_weight_name, "w1", expert_id\n'
    "                        )\n"
    "                        continue\n"
    '                    param_name = f"{fused_moe_prefix}.w13_weight"\n'
    "                    if param_name not in params_dict:\n"
    '                        param_name_packed = f"{param_name}_packed"\n'
    "                        if param_name_packed in params_dict:\n"
    "                            param_name = param_name_packed\n"
    "                    param = params_dict[param_name]"
)

if old1 in content:
    content = content.replace(old1, new1)
    print("Fix 3 applied: w13 scale routing")
else:
    print("Fix 3: pattern not found")

# Fix for linear_fc2 (w2) scale routing
old2 = (
    '                elif parts[-2] == "linear_fc2":\n'
    '                    param_name = f"{fused_moe_prefix}.w2_weight"\n'
    "                    if param_name not in params_dict:\n"
    '                        param_name_packed = f"{param_name}_packed"\n'
    "                        if param_name_packed in params_dict:\n"
    "                            param_name = param_name_packed\n"
    "                    param = params_dict[param_name]"
)
new2 = (
    '                elif parts[-2] == "linear_fc2":\n'
    "                    # Route weight_scale keys to scale param\n"
    '                    if "weight_scale" in chkpt_weight_name:\n'
    '                        scale_param_name = f"{fused_moe_prefix}.w2_weight_scale"\n'
    "                        if scale_param_name not in params_dict:\n"
    '                            scale_param_name_packed = f"{scale_param_name}_packed"\n'
    "                            if scale_param_name_packed in params_dict:\n"
    "                                scale_param_name = scale_param_name_packed\n"
    "                        scale_param = params_dict[scale_param_name]\n"
    "                        fused_moe_module.weight_loader(\n"
    '                            scale_param, loaded_weight, chkpt_weight_name, "w2", expert_id\n'
    "                        )\n"
    "                        continue\n"
    '                    param_name = f"{fused_moe_prefix}.w2_weight"\n'
    "                    if param_name not in params_dict:\n"
    '                        param_name_packed = f"{param_name}_packed"\n'
    "                        if param_name_packed in params_dict:\n"
    "                            param_name = param_name_packed\n"
    "                    param = params_dict[param_name]"
)

if old2 in content:
    content = content.replace(old2, new2)
    print("Fix 4 applied: w2 scale routing")
else:
    print("Fix 4: pattern not found")

with open(path, "w") as f:
    f.write(content)
print("zaya.py scale routing updated.")
