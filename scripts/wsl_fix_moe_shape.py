p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
with open(p) as f:
    c = f.read()

old = """                    if loaded_weight.ndim == 1:
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
                        )"""

new = """                    if loaded_weight.ndim == 1 or loaded_weight.dtype == torch.uint8 or loaded_weight.dtype == torch.int8:
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
                        )"""

if old in c:
    c = c.replace(old, new)
    with open(p, "w") as f:
        f.write(c)
    print("OK")
else:
    print("Pattern not found. Searching for alternatives...")
    if "loaded_weight.ndim == 1" in c:
        print("Found ndim check but full pattern differs")
    else:
        print("ndim check not present")
