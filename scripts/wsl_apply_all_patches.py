"""Apply all patches needed for Zaya CT NVFP4 vLLM serving in WSL2.

Patches needed:
1. [PRE-APPLIED] Zyphra Python overlay (5 files) onto stock vLLM 0.20.2
2. [PRE-APPLIED] ModelRegistry: register ZayaForCausalLM
3. [PRE-APPLIED] MambaState calculators: cca_state_shape + cca_state_dtype
4. [NEW] zaya.py load_weights: handle compressed_tensors MoE naming
5. [NEW] zaya.py f-string: fix {chkpt_weight_name} format bug
"""

import os
import sys

VLLM = "/root/vllm-env/lib/python3.12/site-packages/vllm"
SRC = "/tmp/zaya-vllm/vllm"
ZAYA_PY = os.path.join(VLLM, "model_executor", "models", "zaya.py")

def apply_patch():
    # 1. Overlay Zyphra fork files
    files = [
        ("model_executor/layers/mamba/cca.py",),
        ("v1/attention/backends/cca_attn.py",),
        ("model_executor/models/zaya.py",),
        ("tool_parsers/zaya_tool_parser.py",),
        ("transformers_utils/configs/zaya.py",),
    ]
    
    for (rel_path,) in files:
        src = os.path.join(SRC, rel_path)
        dst = os.path.join(VLLM, rel_path)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(src, 'rb') as f:
                content = f.read()
            with open(dst, 'wb') as f:
                f.write(content)
            print(f"  Overlaid: {rel_path}")
    
    # 2. Register ZayaForCausalLM
    registry = os.path.join(VLLM, "model_executor", "models", "registry.py")
    with open(registry) as f:
        r = f.read()
    if '"ZayaForCausalLM"' not in r:
        r = r.replace('_TEXT_GENERATION_MODELS = {', '_TEXT_GENERATION_MODELS = {\n    "ZayaForCausalLM": ("zaya", "ZayaForCausalLM"),')
        with open(registry, 'w') as f:
            f.write(r)
        print("  Patched: ModelRegistry - ZayaForCausalLM registered")
    
    # 3. Fix zaya.py load_weights for compressed_tensors
    with open(ZAYA_PY) as f:
        z = f.read()
    
    patches_applied = 0
    
    # Fix f-string
    if '{chkpt_weight_name}' in z:
        z = z.replace(
            'WARNING: key {chkpt_weight_name} not in params!',
            'WARNING: key %s not in params!'
        )
        patches_applied += 1
        print("  Patched: zaya.py f-string format bug")
    
    # Fix MoE: try _packed variant first for compressed_tensors  
    old_moe = 'param_name = f"{fused_moe_prefix}.w13_weight"\n                    param = params_dict[param_name]'
    new_moe = 'param_name = f"{fused_moe_prefix}.w13_weight"\n                    if param_name not in params_dict:\n                        param_name_packed = f"{param_name}_packed"\n                        if param_name_packed in params_dict:\n                            param_name = param_name_packed\n                    param = params_dict[param_name]'
    if old_moe in z:
        z = z.replace(old_moe, new_moe)
        patches_applied += 1
        print("  Patched: zaya.py w13_weight CT support")
    
    old_moe2 = 'param_name = f"{fused_moe_prefix}.w2_weight"\n                    param = params_dict[param_name]'
    new_moe2 = 'param_name = f"{fused_moe_prefix}.w2_weight"\n                    if param_name not in params_dict:\n                        param_name_packed = f"{param_name}_packed"\n                        if param_name_packed in params_dict:\n                            param_name = param_name_packed\n                    param = params_dict[param_name]'
    if old_moe2 in z:
        z = z.replace(old_moe2, new_moe2)
        patches_applied += 1
        print("  Patched: zaya.py w2_weight CT support")
    
    # Fix: use %s instead of f-string for chkpt_weight_name in log  
    old_log = 'logger.info(\n                    "WARNING: key %s not in params! Skipping loading"'
    if old_log in z:
        # Already fixed above
        pass
    
    # Fix the variable reference in the log
    old_log2 = '"WARNING: key %s not in params! Skipping loading"'
    if old_log2 in z:
        # Need to add % chkpt_weight_name
        z = z.replace(
            old_log2 + '\n                    ',
            '"WARNING: key %s not in params! Skipping loading" % chkpt_weight_name\n                    '
        )
        patches_applied += 1
        print("  Patched: zaya.py log format fix")
    
    with open(ZAYA_PY, 'w') as f:
        f.write(z)
    
    print(f"\nTotal patches applied: {patches_applied}")
    return 0

if __name__ == "__main__":
    sys.exit(apply_patch())
