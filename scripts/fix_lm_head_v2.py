"""Apply lm_head fix to zaya.py."""

path = "/home/ttimm/vllm-src/vllm/model_executor/models/zaya.py"
with open(path) as f:
    content = f.read()


# Fix 1: Add lm_head collector before loop
old_loop = "        for chkpt_weight_name, loaded_weight in tqdm.tqdm("
new_loop = (
    "        lm_head_buffers = {}  # collect lm_head NVFP4 params for tied embedding dequant\n"
    "        for chkpt_weight_name, loaded_weight in tqdm.tqdm("
)
if old_loop in content:
    content = content.replace(old_loop, new_loop)
    print("Added lm_head collector init")
else:
    print("ERROR: loop pattern not found")


# Fix 2: Collect lm_head keys
old_skip = (
    "            if chkpt_weight_name not in params_dict:\n"
    "                logger.info(\n"
    '                    f"WARNING: key {chkpt_weight_name} not in params! Skipping loading"\n'
    "                )\n"
    "                continue"
)
new_skip = (
    "            if chkpt_weight_name not in params_dict:\n"
    "                # Collect lm_head NVFP4 weights for tied-embedding dequant\n"
    '                if "lm_head." in chkpt_weight_name:\n'
    "                    lm_head_buffers[chkpt_weight_name] = loaded_weight\n"
    "                    continue\n"
    "                logger.info(\n"
    '                    f"WARNING: key {chkpt_weight_name} not in params! Skipping loading"\n'
    "                )\n"
    "                continue"
)
if old_skip in content:
    content = content.replace(old_skip, new_skip)
    print("Added lm_head collector in skip")
else:
    print("ERROR: skip pattern not found")


# Fix 3: Add dequant before return
old_ret = "        return loaded_params"
new_ret = (
    "        # Dequantize lm_head NVFP4 weights into tied embed_tokens\n"
    "        if lm_head_buffers:\n"
    "            self._dequant_lm_head(lm_head_buffers)\n"
    "        return loaded_params"
)
if old_ret in content:
    content = content.replace(old_ret, new_ret)
    print("Added dequant finalization")
else:
    print("ERROR: return pattern not found")


# Fix 4: Add _dequant_lm_head method after class def
old_cls = "class ZayaForCausalLM(nn.Module, HasInnerState, IsHybrid):"
new_cls = """class ZayaForCausalLM(nn.Module, HasInnerState, IsHybrid):


    def _dequant_lm_head(self, buffers: dict) -> None:
        \"\"\"Dequantize lm_head NVFP4 weights into tied embed_tokens.weight.\"\"\"
        import torch


        pk = [k for k in buffers if "weight_packed" in k]
        sk = [k for k in buffers if "weight_scale" in k]
        if not pk:
            return


        packed = buffers[pk[0]].to(self.model.embed_tokens.weight.device)
        w_lo = (packed & 0x0F).float()
        w_hi = ((packed >> 4) & 0x0F).float()


        def e2m1(x):
            s = -((x >> 3) & 1).float() * 2 + 1
            e = ((x >> 1) & 3).float()
            m = (x & 1).float()
            v = s * (2.0 ** (e - 2.0)) * (1.0 + m / 2.0)
            return torch.where(x == 0, torch.zeros_like(v), v)


        w_lo = e2m1(w_lo)
        w_hi = e2m1(w_hi)
        o, h = packed.shape
        w = torch.zeros(o, h * 2, dtype=w_lo.dtype, device=packed.device)
        w[:, 0::2] = w_lo
        w[:, 1::2] = w_hi


        if sk:
            ws = buffers[sk[0]].float().to(w.device)
            ws = ws.unsqueeze(-1).repeat(1, 1, 16).reshape(o, h * 2)
            w = w * ws


        w = w.to(dtype=self.model.embed_tokens.weight.dtype)
        self.model.embed_tokens.weight.data.copy_(w)
        logger.info("Dequantized lm_head NVFP4 -> embed_tokens.weight shape=%s", list(w.shape))
"""
if old_cls in content:
    content = content.replace(old_cls, new_cls)
    print("Added _dequant_lm_head method")
else:
    print("ERROR: class pattern not found")


with open(path, "w") as f:
    f.write(content)
print("Done.")
