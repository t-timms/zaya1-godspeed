#!/usr/bin/env python3
"""Final fix: add mlp. fallback to BOTH fused_moe module lookup AND param lookup."""

p = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
c = open(p).read()

# Add mlp. fallback to fused_moe_modules.get()
c = c.replace(
    "                fused_moe_module = fused_moe_modules.get(fused_moe_prefix)\n                if fused_moe_module is None:",
    "                fused_moe_module = fused_moe_modules.get(fused_moe_prefix)\n                if fused_moe_module is None:\n                    alt_prefix = fused_moe_prefix.replace('.zaya_block.', '.mlp.zaya_block.')\n                    fused_moe_module = fused_moe_modules.get(alt_prefix)\n                if fused_moe_module is None:",
)

open(p, "w").write(c)
print("ADDED mlp fallback to fused_moe_module lookup")
