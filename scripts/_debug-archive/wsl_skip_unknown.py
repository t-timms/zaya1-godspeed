#!/usr/bin/env python3
"""Production fix: skip unknown weights instead of crashing."""

p = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
c = open(p).read()

# Replace the raise KeyError in local_experts branch with skip+warning
c = c.replace(
    '                    if param is None:\n                        raise KeyError(f"{param_name}")\n                    half',
    '                    if param is None:\n                        logger.warning(\n                            "Unknown param %s, skipping weight %s",\n                            param_name, chkpt_weight_name,\n                        )\n                        continue\n                    half',
)
c = c.replace(
    '                    if param is None:\n                        raise KeyError(f"{param_name}")\n                    fused_moe_module.weight_loader(',
    '                    if param is None:\n                        logger.warning(\n                            "Unknown param %s, skipping weight %s",\n                            param_name, chkpt_weight_name,\n                        )\n                        continue\n                    fused_moe_module.weight_loader(',
)
c = c.replace(
    '            if param is None:\n                raise KeyError(f"{chkpt_weight_name}")',
    '            if param is None:\n                logger.warning(\n                    "Unknown weight key %s, skipping", chkpt_weight_name\n                )\n                continue',
)

open(p, "w").write(c)
print("SKIP-INSTEAD-OF-CRASH applied")
