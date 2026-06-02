"""Apply MoE weight loading patches to Zyphra fork's zaya.py."""

p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
with open(p) as f:
    c = f.read()

# Fix 1: w13_weight - add _packed fallback
old = '                if parts[-2] == "linear_fc1":\n                    param_name = f"{fused_moe_prefix}.w13_weight"\n                    param = params_dict[param_name]\n                    half = loaded_weight.shape[0] // 2'
new = """                if parts[-2] == "linear_fc1":
                    param_name = f"{fused_moe_prefix}.w13_weight"
                    if param_name not in params_dict:
                        param_name = f"{param_name}_packed"
                    if param_name not in params_dict:
                        logger.warning("No w13_weight at %s, skipping %s", fused_moe_prefix, chkpt_weight_name)
                        continue
                    param = params_dict[param_name]
                    # Convert uint8 [out, in//2] to int32 [in//8, out] for CT MoE
                    if loaded_weight.dtype in (torch.uint8, torch.int8):
                        loaded_weight = loaded_weight.to(torch.int32).t().contiguous()
                    half = loaded_weight.shape[0] // 2"""
c = c.replace(old, new)

# Fix 2: w2_weight - add _packed fallback
old2 = '                elif parts[-2] == "linear_fc2":\n                    param_name = f"{fused_moe_prefix}.w2_weight"\n                    param = params_dict[param_name]'
new2 = """                elif parts[-2] == "linear_fc2":
                    param_name = f"{fused_moe_prefix}.w2_weight"
                    if param_name not in params_dict:
                        param_name = f"{param_name}_packed"
                    if param_name not in params_dict:
                        logger.warning("No w2_weight at %s, skipping %s", fused_moe_prefix, chkpt_weight_name)
                        continue
                    param = params_dict[param_name]
                    if loaded_weight.dtype in (torch.uint8, torch.int8):
                        loaded_weight = loaded_weight.to(torch.int32).t().contiguous()"""
c = c.replace(old2, new2)

# Fix 3: Add torch import if missing
if "import torch" not in c[:200]:
    c = c.replace("from collections import", "import torch\nfrom collections import")

with open(p, "w") as f:
    f.write(c)
print("Fork zaya.py patched: _packed fallback + uint8->int32 conversion")
