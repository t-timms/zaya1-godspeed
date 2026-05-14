#!/usr/bin/env python3
"""Fix: for CT-packed MoE weights, pass full w13 without splitting."""

p = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
c = open(p).read()

# Fix linear_fc1: if param is _packed, don't split - pass full weight
c = c.replace(
    '                    half = loaded_weight.shape[0] // 2\n                    gate_weight = loaded_weight[:half, :]\n                    up_weight = loaded_weight[half:, :]\n                    fused_moe_module.weight_loader(\n                        param, gate_weight, chkpt_weight_name, "w1", expert_id\n                    )\n                    fused_moe_module.weight_loader(\n                        param, up_weight, chkpt_weight_name, "w3", expert_id\n                    )\n                    loaded_params.add(param_name)',
    '                    if "_packed" in param_name:\n                        fused_moe_module.weight_loader(\n                            param, loaded_weight, chkpt_weight_name, "w1", expert_id\n                        )\n                    else:\n                        half = loaded_weight.shape[0] // 2\n                        gate_weight = loaded_weight[:half, :]\n                        up_weight = loaded_weight[half:, :]\n                        fused_moe_module.weight_loader(\n                            param, gate_weight, chkpt_weight_name, "w1", expert_id\n                        )\n                        fused_moe_module.weight_loader(\n                            param, up_weight, chkpt_weight_name, "w3", expert_id\n                        )\n                    loaded_params.add(param_name)',
)

# Fix linear_fc2: same for w2
c = c.replace(
    '                    fused_moe_module.weight_loader(\n                        param, loaded_weight, chkpt_weight_name, "w2", expert_id\n                    )\n                    loaded_params.add(param_name)',
    '                    shard_id = None if "_packed" in param_name else "w2"\n                    fused_moe_module.weight_loader(\n                        param, loaded_weight, chkpt_weight_name, shard_id, expert_id\n                    )\n                    loaded_params.add(param_name)',
)

open(p, "w").write(c)
print("FIXED: CT-packed MoE weights loaded without splitting")
