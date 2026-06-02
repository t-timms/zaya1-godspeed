"""Fix all remaining compressed_tensors.py and zaya.py issues for NVFP4A16."""

# ── Fix 1: compressed_tensors.py - add input_quant None guards ──
ct_path = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
with open(ct_path) as f:
    ct = f.read()

fixes = 0

# Fix _is_static_tensor_w8a8: add early return + fix line 439, 441
old = """        is_8_bits = weight_quant.num_bits == 8 and input_quant is not None and input_quant.num_bits == 8
        if not is_8_bits:
            return False

        weight_strategy = weight_quant.strategy == QuantizationStrategy.TENSOR.value
        input_strategy = input_quant.strategy == QuantizationStrategy.TENSOR.value
        if not weight_strategy or not input_strategy:
            return False

        is_static = not weight_quant.dynamic and not input_quant.dynamic
        return is_static"""
new = """        if input_quant is None:
            return False
        is_8_bits = weight_quant.num_bits == 8 and input_quant.num_bits == 8
        if not is_8_bits:
            return False

        weight_strategy = weight_quant.strategy == QuantizationStrategy.TENSOR.value
        input_strategy = input_quant.strategy == QuantizationStrategy.TENSOR.value
        if not weight_strategy or not input_strategy:
            return False

        is_static = not weight_quant.dynamic and not input_quant.dynamic
        return is_static"""
if old in ct:
    ct = ct.replace(old, new)
    fixes += 1
    print("  Fixed: _is_static_tensor_w8a8")

# Fix _is_dynamic_token_w8a8: add early return
old = """        is_8_bits = weight_quant.num_bits == 8 and input_quant.num_bits == 8
        if not is_8_bits:
            return False

        weight_strategy = weight_quant.strategy == QuantizationStrategy.TOKEN.value
        input_strategy = input_quant.strategy == QuantizationStrategy.TOKEN.value
        if not weight_strategy or not input_strategy:
            return False

        weight_dynamic = weight_quant.dynamic
        input_dynamic = input_quant.dynamic"""
new = """        if input_quant is None:
            return False
        is_8_bits = weight_quant.num_bits == 8 and input_quant.num_bits == 8
        if not is_8_bits:
            return False

        weight_strategy = weight_quant.strategy == QuantizationStrategy.TOKEN.value
        input_strategy = input_quant.strategy == QuantizationStrategy.TOKEN.value
        if not weight_strategy or not input_strategy:
            return False

        weight_dynamic = weight_quant.dynamic
        input_dynamic = input_quant.dynamic"""
if old in ct:
    ct = ct.replace(old, new)
    fixes += 1
    print("  Fixed: _is_dynamic_token_w8a8")

# Fix _is_dynamic_token_w4a8_int: add early return
old = """        is_8_bits = input_quant.num_bits == 8
        if not is_8_bits:"""
new = """        if input_quant is None:
            return False
        is_8_bits = input_quant.num_bits == 8
        if not is_8_bits:"""
if old in ct:
    ct = ct.replace(old, new)
    fixes += 1
    print("  Fixed: _is_dynamic_token_w4a8_int")

with open(ct_path, "w") as f:
    f.write(ct)

# ── Fix 2: zaya.py - KeyError guards and log fix ──
zaya_path = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
with open(zaya_path) as f:
    za = f.read()

# Fix w13_weight KeyError: add warning + continue if both variants missing
old_moe = """                    param = params_dict[param_name]
                    if loaded_weight.ndim == 1:
                        fused_moe_module.weight_loader(
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
                        )
                    loaded_params.add(param_name)"""
new_moe = """                    if param_name not in params_dict:
                        logger.warning(
                            "No w13_weight or w13_weight_packed at %s, skipping %s",
                            fused_moe_prefix, chkpt_weight_name
                        )
                        continue
                    param = params_dict[param_name]
                    if loaded_weight.ndim == 1:
                        fused_moe_module.weight_loader(
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
                        )
                    loaded_params.add(param_name)"""
if old_moe in za:
    za = za.replace(old_moe, new_moe)
    fixes += 1
    print("  Fixed: zaya.py w13_weight KeyError guard")

# Fix w2_weight KeyError: add warning + continue
old_moe2 = """                    param = params_dict[param_name]
                    fused_moe_module.weight_loader(
                        param, loaded_weight, chkpt_weight_name, "w2", expert_id
                    )
                    loaded_params.add(param_name)"""
new_moe2 = """                    if param_name not in params_dict:
                        logger.warning(
                            "No w2_weight or w2_weight_packed at %s, skipping %s",
                            fused_moe_prefix, chkpt_weight_name
                        )
                        continue
                    param = params_dict[param_name]
                    fused_moe_module.weight_loader(
                        param, loaded_weight, chkpt_weight_name, "w2", expert_id
                    )
                    loaded_params.add(param_name)"""
if old_moe2 in za:
    za = za.replace(old_moe2, new_moe2)
    fixes += 1
    print("  Fixed: zaya.py w2_weight KeyError guard")

# Fix logger missing argument
old_log = """            logger.info(
                    "WARNING: key %s not in params! Skipping loading"
                )"""
new_log = """            logger.info(
                    "WARNING: key %s not in params! Skipping loading",
                    chkpt_weight_name,
                )"""
if old_log in za:
    za = za.replace(old_log, new_log)
    fixes += 1
    print("  Fixed: zaya.py logger missing arg")

with open(zaya_path, "w") as f:
    f.write(za)

print(f"\nTotal fixes applied: {fixes}")
