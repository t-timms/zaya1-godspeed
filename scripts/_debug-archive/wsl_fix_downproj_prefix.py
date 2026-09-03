#!/usr/bin/env python3
"""Fix zaya.py: pass prefix to router's down_proj ReplicatedLinear."""

path = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
content = open(path).read()

old = """        self.down_proj = ReplicatedLinear(
            self.hidden_size,
            self.mlp_expansion,
            bias=True,
            quant_config=quant_config,
            return_bias=False,
        )"""

new = """        self.down_proj = ReplicatedLinear(
            self.hidden_size,
            self.mlp_expansion,
            bias=True,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.down_proj",
        )"""

if 'prefix=f"{prefix}.down_proj"' not in content:
    content = content.replace(old, new)
    open(path, "w").write(content)
    print("ADDED down_proj prefix")
else:
    print("ALREADY PRESENT")
