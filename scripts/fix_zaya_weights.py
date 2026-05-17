"""Apply fixes to zaya.py for NVFP4 weight loading."""
path = "/home/ttimm/vllm-src/vllm/model_executor/models/zaya.py"
with open(path) as f:
    content = f.read()

# Fix 1: Check for _packed variant in w13_weight lookup
old = (
    '                if parts[-2] == "linear_fc1":\n'
    '                    param_name = f"{fused_moe_prefix}.w13_weight"\n'
    '                    param = params_dict[param_name]'
)
new = (
    '                if parts[-2] == "linear_fc1":\n'
    '                    param_name = f"{fused_moe_prefix}.w13_weight"\n'
    '                    if param_name not in params_dict:\n'
    '                        param_name_packed = f"{param_name}_packed"\n'
    '                        if param_name_packed in params_dict:\n'
    '                            param_name = param_name_packed\n'
    '                    param = params_dict[param_name]'
)

if old in content:
    content = content.replace(old, new)
    print("Fix 1 applied: w13_weight _packed variant check")
else:
    print("Fix 1: pattern not found")

# Fix 2: Check for _packed variant in w2_weight lookup
old2 = (
    '                    param_name = f"{fused_moe_prefix}.w2_weight"\n'
    '                    param = params_dict[param_name]'
)
new2 = (
    '                    param_name = f"{fused_moe_prefix}.w2_weight"\n'
    '                    if param_name not in params_dict:\n'
    '                        param_name_packed = f"{param_name}_packed"\n'
    '                        if param_name_packed in params_dict:\n'
    '                            param_name = param_name_packed\n'
    '                    param = params_dict[param_name]'
)

if old2 in content:
    content = content.replace(old2, new2)
    print("Fix 2 applied: w2_weight _packed variant check")
else:
    print("Fix 2: pattern not found")

with open(path, "w") as f:
    f.write(content)
print("zaya.py updated.")
