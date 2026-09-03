"""Patch zaya.py load_weights to convert uint8 packed MoE weights to int32 layout
expected by CompressedTensorsWNA16MoEMethod.

Our format: [out, in//2] uint8 (2 fp4 per byte)
CT MoE format: [in//8, out] int32 (8 fp4 per int32), then transposed later
"""

p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
with open(p) as f:
    c = f.read()

# Insert conversion for MoE packed weights before weight_loader call
# Find the w13/w2 weight_loader call and add conversion before it
old = """                if param_name not in params_dict:
                        logger.warning(
                            "No w13_weight or w13_weight_packed at %s, skipping %s",
                            fused_moe_prefix, chkpt_weight_name
                        )
                        continue
                    param = params_dict[param_name]
                    if loaded_weight.ndim == 1 or loaded_weight.dtype == torch.uint8 or loaded_weight.dtype == torch.int8:
                        fused_moe_module.weight_loader(
                            param, loaded_weight, chkpt_weight_name, "w1", expert_id
                        )
                    else:"""

new = """                if param_name not in params_dict:
                        logger.warning(
                            "No w13_weight or w13_weight_packed at %s, skipping %s",
                            fused_moe_prefix, chkpt_weight_name
                        )
                        continue
                    # Convert uint8 packed [out, in//2] to int32 [in//8, out] for CT MoE
                    if loaded_weight.dtype == torch.uint8 and loaded_weight.ndim == 2:
                        int32_size = loaded_weight.numel() // 4
                        loaded_weight = loaded_weight.to("cuda:0").view(torch.int32).view(
                            -1, loaded_weight.shape[0]
                        ).t().contiguous()
                    param = params_dict[param_name]
                    if loaded_weight.ndim == 1 or loaded_weight.dtype == torch.uint8 or loaded_weight.dtype == torch.int8:
                        fused_moe_module.weight_loader(
                            param, loaded_weight, chkpt_weight_name, "w1", expert_id
                        )
                    else:"""

if old in c:
    c = c.replace(old, new)
    with open(p, "w") as f:
        f.write(c)
    print("Patched: uint8->int32 MoE weight conversion for w13")
else:
    print("Pattern not found for w13. Searching...")
    if "No w13_weight or w13_weight_packed" in c:
        print("Found w13 guard, but surrounding pattern differs")
    else:
        print("w13 guard not found")

# Also fix w2 similarly
old2 = """                if param_name not in params_dict:
                        logger.warning(
                            "No w2_weight or w2_weight_packed at %s, skipping %s",
                            fused_moe_prefix, chkpt_weight_name
                        )
                        continue
                    param = params_dict[param_name]
                    fused_moe_module.weight_loader(
                        param, loaded_weight, chkpt_weight_name, "w2", expert_id
                    )"""

new2 = """                if param_name not in params_dict:
                        logger.warning(
                            "No w2_weight or w2_weight_packed at %s, skipping %s",
                            fused_moe_prefix, chkpt_weight_name
                        )
                        continue
                    # Convert uint8 packed [out, in//2] to int32 [in//8, out] for CT MoE
                    if loaded_weight.dtype == torch.uint8 and loaded_weight.ndim == 2:
                        int32_size = loaded_weight.numel() // 4
                        loaded_weight = loaded_weight.to("cuda:0").view(torch.int32).view(
                            -1, loaded_weight.shape[0]
                        ).t().contiguous()
                    param = params_dict[param_name]
                    fused_moe_module.weight_loader(
                        param, loaded_weight, chkpt_weight_name, "w2", expert_id
                    )"""

if old2 in c:
    c = c.replace(old2, new2)
    with open(p, "w") as f:
        f.write(c)
    print("Patched: uint8->int32 MoE weight conversion for w2")
else:
    print("Pattern not found for w2")
