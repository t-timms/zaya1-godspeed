import re

with open('/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py') as f:
    z = f.read()

old = '                if parts[-2] == "linear_fc1":\n                    param_name = f"{fused_moe_prefix}.w13_weight"\n                    if param_name not in params_dict:\n                        param_name_packed = f"{param_name}_packed"\n                        if param_name_packed in params_dict:\n                            param_name = param_name_packed\n                    param = params_dict[param_name]\n                    half = loaded_weight.shape[0] // 2\n                    gate_weight = loaded_weight[:half, :]\n                    up_weight = loaded_weight[half:, :]\n                    fused_moe_module.weight_loader(\n                        param, gate_weight, chkpt_weight_name, "w1", expert_id\n                    )\n                    fused_moe_module.weight_loader(\n                        param, up_weight, chkpt_weight_name, "w3", expert_id\n                    )\n                    loaded_params.add(param_name)'

new = '                if parts[-2] == "linear_fc1":\n                    param_name = f"{fused_moe_prefix}.w13_weight"\n                    if param_name not in params_dict:\n                        param_name_packed = f"{param_name}_packed"\n                        if param_name_packed in params_dict:\n                            param_name = param_name_packed\n                    param = params_dict[param_name]\n                    if loaded_weight.ndim == 1:\n                        fused_moe_module.weight_loader(\n                            param, loaded_weight, chkpt_weight_name, "w1", expert_id\n                        )\n                    else:\n                        half = loaded_weight.shape[0] // 2\n                        gate_weight = loaded_weight[:half, :]\n                        up_weight = loaded_weight[half:, :]\n                        fused_moe_module.weight_loader(\n                            param, gate_weight, chkpt_weight_name, "w1", expert_id\n                        )\n                        fused_moe_module.weight_loader(\n                            param, up_weight, chkpt_weight_name, "w3", expert_id\n                        )\n                    loaded_params.add(param_name)'

assert old in z, "Pattern not found in zaya.py!"
z = z.replace(old, new)
with open('/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py', 'w') as f:
    f.write(z)
print('OK - 1D weight handling patched')
