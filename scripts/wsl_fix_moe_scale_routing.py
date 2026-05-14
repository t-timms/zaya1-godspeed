#!/usr/bin/env python3
"""Fix: Route weight_scale checkpoint keys to correct FusedMoE scale params.

Bug: zaya.py load_weights routes `...linear_fc1.weight_scale` to `w13_weight`
param instead of `w13_weight_scale` param. The FusedMoE's _load_weight
checks quant_method on the param, but w13_weight has no quant_method,
causing ValueError at fused_moe/layer.py:1359.

Fix: In the linear_fc1/linear_fc2 branches, detect weight_scale suffix
and route to w13_weight_scale / w2_weight_scale respectively.
"""

from pathlib import Path

ZAYA_PY = Path(
    "/home/ttimm/vllm-env/lib/python3.12/site-packages/"
    "vllm/model_executor/models/zaya.py"
)


def fix() -> bool:
    content = ZAYA_PY.read_text()
    modified = False

    # ── Fix 1: linear_fc1 branch ──
    # Old: always routes to w13_weight, then weight_loader splits gate/up
    # New: check if weight_scale → route to w13_weight_scale; else existing
    old_fc1 = (
        '                if parts[-2] == "linear_fc1":\n'
        '                    param_name = f"{fused_moe_prefix}.w13_weight"\n'
        '                    param = params_dict.get(param_name)\n'
        '                    if param is None:\n'
        '                        param_name = f"{fused_moe_prefix}.w13_weight_packed"\n'
        '                        param = params_dict.get(param_name)\n'
        '                    if param is None:\n'
        '                        param_name = param_name.replace(".zaya_block.", ".mlp.zaya_block.")\n'
        '                        param = params_dict.get(param_name)\n'
        '                    if param is None:\n'
        '                        raise KeyError(f"FusedMoE w13 param not found for {fused_moe_prefix}")\n'
    )
    new_fc1 = (
        '                if parts[-2] == "linear_fc1":\n'
        '                    if "weight_scale" in chkpt_weight_name:\n'
        '                        param_name = f"{fused_moe_prefix}.w13_weight_scale"\n'
        '                        param = params_dict.get(param_name)\n'
        '                        if param is None:\n'
        '                            param_name = param_name.replace(".zaya_block.", ".mlp.zaya_block.")\n'
        '                            param = params_dict.get(param_name)\n'
        '                        if param is None:\n'
        '                            raise KeyError(f"FusedMoE w13_weight_scale not found for {fused_moe_prefix}")\n'
        '                        fused_moe_module.weight_loader(\n'
        '                            param, loaded_weight, chkpt_weight_name, "w1", expert_id\n'
        '                        )\n'
        '                        loaded_params.add(param_name)\n'
        '                        continue\n'
        '                    param_name = f"{fused_moe_prefix}.w13_weight"\n'
        '                    param = params_dict.get(param_name)\n'
        '                    if param is None:\n'
        '                        param_name = f"{fused_moe_prefix}.w13_weight_packed"\n'
        '                        param = params_dict.get(param_name)\n'
        '                    if param is None:\n'
        '                        param_name = param_name.replace(".zaya_block.", ".mlp.zaya_block.")\n'
        '                        param = params_dict.get(param_name)\n'
        '                    if param is None:\n'
        '                        raise KeyError(f"FusedMoE w13 param not found for {fused_moe_prefix}")\n'
    )

    if old_fc1 in content:
        content = content.replace(old_fc1, new_fc1)
        print("  [FIX] linear_fc1: added weight_scale routing → w13_weight_scale")
        modified = True
    else:
        # Check if already patched
        if 'weight_scale" in chkpt_weight_name' in content:
            print("  [OK] linear_fc1: weight_scale routing already present")
        else:
            print("  [WARN] linear_fc1: pattern not found — may need manual fix")

    # ── Fix 2: linear_fc2 branch ──
    # Old: always routes to w2_weight
    # New: check if weight_scale → route to w2_weight_scale; else existing
    old_fc2 = (
        '                elif parts[-2] == "linear_fc2":\n'
        '                    param_name = f"{fused_moe_prefix}.w2_weight"\n'
        '                    param = params_dict.get(param_name)\n'
        '                    if param is None:\n'
        '                        param_name = f"{fused_moe_prefix}.w2_weight_packed"\n'
        '                        param = params_dict.get(param_name)\n'
        '                    if param is None:\n'
        '                        param_name = param_name.replace(".zaya_block.", ".mlp.zaya_block.")\n'
        '                        param = params_dict.get(param_name)\n'
        '                    if param is None:\n'
        '                        raise KeyError(f"FusedMoE w2 param not found for {fused_moe_prefix}")\n'
    )
    new_fc2 = (
        '                elif parts[-2] == "linear_fc2":\n'
        '                    if "weight_scale" in chkpt_weight_name:\n'
        '                        param_name = f"{fused_moe_prefix}.w2_weight_scale"\n'
        '                        param = params_dict.get(param_name)\n'
        '                        if param is None:\n'
        '                            param_name = param_name.replace(".zaya_block.", ".mlp.zaya_block.")\n'
        '                            param = params_dict.get(param_name)\n'
        '                        if param is None:\n'
        '                            raise KeyError(f"FusedMoE w2_weight_scale not found for {fused_moe_prefix}")\n'
        '                        fused_moe_module.weight_loader(\n'
        '                            param, loaded_weight, chkpt_weight_name, "w2", expert_id\n'
        '                        )\n'
        '                        loaded_params.add(param_name)\n'
        '                        continue\n'
        '                    param_name = f"{fused_moe_prefix}.w2_weight"\n'
        '                    param = params_dict.get(param_name)\n'
        '                    if param is None:\n'
        '                        param_name = f"{fused_moe_prefix}.w2_weight_packed"\n'
        '                        param = params_dict.get(param_name)\n'
        '                    if param is None:\n'
        '                        param_name = param_name.replace(".zaya_block.", ".mlp.zaya_block.")\n'
        '                        param = params_dict.get(param_name)\n'
        '                    if param is None:\n'
        '                        raise KeyError(f"FusedMoE w2 param not found for {fused_moe_prefix}")\n'
    )

    if old_fc2 in content:
        content = content.replace(old_fc2, new_fc2)
        print("  [FIX] linear_fc2: added weight_scale routing → w2_weight_scale")
        modified = True
    else:
        if 'weight_scale" in chkpt_weight_name' in content:
            print("  [OK] linear_fc2: weight_scale routing already present")
        else:
            print("  [WARN] linear_fc2: pattern not found — may need manual fix")

    if modified:
        ZAYA_PY.write_text(content)
        print("  [SAVED] zaya.py updated")

    return modified


if __name__ == "__main__":
    if not ZAYA_PY.exists():
        print(f"ERROR: zaya.py not found at {ZAYA_PY}")
        exit(1)
    ok = fix()
    if ok:
        print("\nFix applied. Re-run smoke test to verify.")
    else:
        print("\nNo changes needed or fix was not applied.")
