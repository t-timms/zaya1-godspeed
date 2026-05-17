#!/usr/bin/env python3
"""Fix: split w13 weight_scale into gate/up halves on load in zaya.py (vllm-src).

The FusedMoE._load_w13 narrows loaded_weight by half for is_act_and_mul=True.
Passing the full [2*N, K//gs] scale with shard_id="w1" only loads gate-half;
up-half scales stay at torch.empty (NaN). This produces NaN dequantized weights.

Split the combined scale tensor into [:half] and [half:] and load as w1/w3.
"""
from pathlib import Path

ZAYA = Path("/home/ttimm/vllm-src/vllm/model_executor/models/zaya.py")
c = ZAYA.read_text()

# The old scale load for linear_fc1 (w13)
old_scale = """                    if "weight_scale" in chkpt_weight_name:
                        scale_param_name = f"{fused_moe_prefix}.w13_weight_scale"
                        if scale_param_name not in params_dict:
                            scale_param_name_packed = f"{scale_param_name}_packed"
                            if scale_param_name_packed in params_dict:
                                scale_param_name = scale_param_name_packed
                        scale_param = params_dict[scale_param_name]
                        fused_moe_module.weight_loader(
                            scale_param, loaded_weight, chkpt_weight_name, "w1", expert_id
                        )
                        continue"""

# The new scale load with gate/up split
new_scale = """                    if "weight_scale" in chkpt_weight_name:
                        scale_param_name = f"{fused_moe_prefix}.w13_weight_scale"
                        if scale_param_name not in params_dict:
                            scale_param_name_packed = f"{scale_param_name}_packed"
                            if scale_param_name_packed in params_dict:
                                scale_param_name = scale_param_name_packed
                        scale_param = params_dict[scale_param_name]
                        # Split combined scale into gate/up halves.
                        # _load_w13 narrows by half for is_act_and_mul=True;
                        # passing the full [2*N, K//gs] tensor with shard_id="w1"
                        # only loads gate-half; up-half stays at torch.empty (NaN).
                        half = loaded_weight.shape[0] // 2
                        fused_moe_module.weight_loader(
                            scale_param, loaded_weight[:half, :], chkpt_weight_name, "w1", expert_id
                        )
                        fused_moe_module.weight_loader(
                            scale_param, loaded_weight[half:, :], chkpt_weight_name, "w3", expert_id
                        )
                        continue"""

if old_scale in c:
    c = c.replace(old_scale, new_scale)
    ZAYA.write_text(c)
    print("  [FIXED] w13 weight_scale split into gate/up halves on load")
elif "Split combined scale into gate/up halves" in c:
    print("  [OK] Scale split already present")
else:
    print("  [WARN] Could not find w13 scale load block")

# Also fix the weight_packed load if it doesn't have the split
old_w13 = """                        fused_moe_module.weight_loader(
                            param, loaded_weight, chkpt_weight_name, "w1", expert_id
                        )
                    else:
                        half = loaded_weight.shape[0] // 2
                        gate_weight = loaded_weight[:half, :]
                        up_weight = loaded_weight[half:, :]
                        fused_moe_module.weight_loader(
                            param, gate_weight, chkpt_weight_name, "w1", expert_id
                        )
                        fused_moe_module.weight_loader(
                            param, up_weight, chkpt_weight_name, "w3", expert_id
                        )"""

if old_w13 in c:
    print("  [OK] w13 weight_packed split already present")
else:
    # Check alternative pattern
    alt = 'fused_moe_module.weight_loader(\n                            param, gate_weight'
    if alt in c:
        print("  [OK] w13 weight_packed has alternate split pattern")
    else:
        print("  [WARN] w13 weight_packed split pattern not found")
