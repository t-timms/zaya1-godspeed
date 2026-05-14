p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
with open(p) as f:
    c = f.read()
old = """                    if is_packed:
                        gate_weight = gate_weight.reshape(-1, 4).view(torch.int32).reshape(gate_weight.shape[0], gate_weight.shape[1] // 4).t().contiguous()
                        up_weight = up_weight.reshape(-1, 4).view(torch.int32).reshape(up_weight.shape[0], up_weight.shape[1] // 4).t().contiguous()"""
new = """                    if is_packed:
                        gate_weight = gate_weight.reshape(-1, 4).view(torch.int32).reshape(gate_weight.shape[0], gate_weight.shape[1] // 4).t().contiguous()
                        up_weight = up_weight.reshape(-1, 4).view(torch.int32).reshape(up_weight.shape[0], up_weight.shape[1] // 4).t().contiguous()
                        logger.warning("DEBUG w13 e%d: gate=%s %s up=%s %s", expert_id, gate_weight.shape, gate_weight.dtype, up_weight.shape, up_weight.dtype)"""
c = c.replace(old, new)
with open(p, "w") as f:
    f.write(c)
print("Debug added")
