"""Fix lm_head NVFP4 dequant for tied embeddings in zaya.py."""

path = "/home/ttimm/vllm-src/vllm/model_executor/models/zaya.py"
with open(path) as f:
    content = f.read()

# Fix 1: Add lm_head collector before the main loop
old_init = "        for chkpt_weight_name, loaded_weight in tqdm.tqdm("
new_init = (
    "        lm_head_buffers = {}  # collect lm_head NVFP4 params for dequant\n"
    "        for chkpt_weight_name, loaded_weight in tqdm.tqdm("
)
if old_init in content:
    content = content.replace(old_init, new_init)
    print("Added lm_head collector init")
else:
    print("lm_head init pattern not found")

# Fix 2: Collect lm_head keys instead of skipping
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
    print("Added lm_head collector in skip logic")
else:
    print("lm_head skip pattern not found")

# Fix 3: Add dequant finalization before return
old_return = "        return loaded_params"
new_return = (
    "        # Dequantize lm_head NVFP4 weights and copy into tied embed_tokens\n"
    "        if lm_head_buffers:\n"
    "            self._dequant_lm_head(lm_head_buffers)\n"
    "        return loaded_params"
)
if old_return in content:
    content = content.replace(old_return, new_return)
    print("Added lm_head dequant finalization")
else:
    print("lm_head return pattern not found")

# Fix 4: Add _dequant_lm_head method to the class
# Find the class definition and add the method
old_class = "class ZayaForCausalLM(nn.Module, SupportsPP):"
new_class = (
    "class ZayaForCausalLM(nn.Module, SupportsPP):\n"
    "    def _dequant_lm_head(self, buffers: dict) -> None:\n"
    '        """Dequantize lm_head NVFP4 weights into tied embed_tokens.weight."""\n'
    "        from compressed_tensors.quantization import QuantizationType\n"
    "        from compressed_tensors.quantization.utils import (\n"
    "            dequantize,\n"
    "        )\n"
    "        import torch\n"
    "        \n"
    '        packed_key = [k for k in buffers if "weight_packed" in k]\n'
    '        scale_key = [k for k in buffers if "weight_scale" in k]\n'
    "        \n"
    "        if not packed_key:\n"
    '            logger.warning("lm_head weight_packed not found in buffers")\n'
    "            return\n"
    "        \n"
    "        packed = buffers[packed_key[0]]\n"
    "        weight_scale = buffers.get(scale_key[0]) if scale_key else None\n"
    "        \n"
    "        # Simple FP4 unpack: 2x 4-bit per uint8 byte, E2M1 format\n"
    "        weight_low = (packed & 0x0F).float()\n"
    "        weight_high = ((packed >> 4) & 0x0F).float()\n"
    "        \n"
    "        # E2M1 decode: s|ee|m -> (-1)^s * 2^(e-2) * (1+m/2)\n"
    "        def e2m1_to_float(x):\n"
    "            sign = -((x >> 3) & 1).float() * 2 + 1  # -1^s\n"
    "            exp = ((x >> 1) & 3).float()\n"
    "            mant = (x & 1).float()\n"
    "            val = sign * (2.0 ** (exp - 2.0)) * (1.0 + mant / 2.0)\n"
    "            val = torch.where(x == 0, torch.zeros_like(val), val)\n"
    "            return val\n"
    "        \n"
    "        weight_low = e2m1_to_float(weight_low)\n"
    "        weight_high = e2m1_to_float(weight_high)\n"
    "        \n"
    "        # Interleave: [out, in/2 * 2] = [out, in]\n"
    "        out_dim, in_half = packed.shape\n"
    "        weight = torch.empty(out_dim, in_half * 2, dtype=weight_low.dtype, device=packed.device)\n"
    "        weight[:, 0::2] = weight_low\n"
    "        weight[:, 1::2] = weight_high\n"
    "        \n"
    "        # Apply per-group scales if available\n"
    "        if weight_scale is not None:\n"
    "            gs = 16  # group_size\n"
    "            weight_scale = weight_scale.float()\n"
    "            weight_scale = weight_scale.unsqueeze(-1).repeat(1, 1, gs).reshape(out_dim, in_half * 2)\n"
    "            weight = weight * weight_scale\n"
    "        \n"
    "        weight = weight.bfloat16()\n"
    "        self.model.embed_tokens.weight.data.copy_(weight)\n"
    '        logger.info("Dequantized lm_head NVFP4 -> embed_tokens.weight shape=%s", list(weight.shape))\n'
)
if old_class in content:
    content = content.replace(old_class, new_class)
    print("Added _dequant_lm_head method")
else:
    print("lm_head class pattern not found")

with open(path, "w") as f:
    f.write(content)
print("zaya.py lm_head fix applied.")
