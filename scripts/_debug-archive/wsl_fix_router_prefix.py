#!/usr/bin/env python3
"""Fix zaya.py: pass prefix to router_mlp ReplicatedLinear layers."""

path = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
content = open(path).read()

old = """            ReplicatedLinear(
                D, D, bias=True, quant_config=quant_config, return_bias=False
            ),
            self.non_linearity,
            ReplicatedLinear(
                D, D, bias=True, quant_config=quant_config, return_bias=False
            ),
            self.non_linearity,
            ReplicatedLinear(
                D, E, bias=False, quant_config=quant_config, return_bias=False
            ),"""

new = """            ReplicatedLinear(
                D, D, bias=True, quant_config=quant_config, return_bias=False,
                prefix=f"{prefix}.router_mlp.0",
            ),
            self.non_linearity,
            ReplicatedLinear(
                D, D, bias=True, quant_config=quant_config, return_bias=False,
                prefix=f"{prefix}.router_mlp.1",
            ),
            self.non_linearity,
            ReplicatedLinear(
                D, E, bias=False, quant_config=quant_config, return_bias=False,
                prefix=f"{prefix}.router_mlp.2",
            ),"""

if "router_mlp.0" not in content:
    content = content.replace(old, new)
    open(path, "w").write(content)
    print("ADDED router_mlp prefixes")
else:
    print("ALREADY PRESENT")
