#!/usr/bin/env python3
"""Fix load_weights: transform CT safetensors keys to match vLLM module hierarchy.

CT-quantized safetensors use: model.layers.N.zaya_block.local_experts.M.linear_fc1
But vLLM's FusedMoE is at: model.layers.N.mlp.zaya_block.experts

Transforms needed:
1. Insert 'mlp.' before 'zaya_block'
2. 'local_experts' -> 'experts'
"""

path = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
content = open(path).read()

# Add helper function
helper = '''
def _ct_map_name(name: str) -> str:
    """Map CT-quantized safetensors names to vLLM module names.

    CT compressor saves weights using vLLM-native naming like:
      model.layers.N.zaya_block.experts.w13_weight
    but vLLM's module hierarchy has an extra 'mlp' level:
      model.layers.N.mlp.zaya_block.experts.w13_weight
    """
    name = name.replace(".zaya_block.local_experts.", ".mlp.zaya_block.experts.")
    name = name.replace(".zaya_block.", ".mlp.zaya_block.")
    return name
'''

if "_ct_map_name" not in content:
    # Insert before load_weights method
    content = content.replace(
        "    def load_weights(",
        helper + "\n    def load_weights(",
    )

# Fix line 925 area - param_name construction for linear_fc1
content = content.replace(
    'param_name = f"{fused_moe_prefix}.w13_weight"',
    'param_name = _ct_map_name(f"{fused_moe_prefix}.w13_weight")',
)

# Fix line 938 area - param_name construction for linear_fc2
content = content.replace(
    'param_name = f"{fused_moe_prefix}.w2_weight"',
    'param_name = _ct_map_name(f"{fused_moe_prefix}.w2_weight")',
)

# Fix line 955 - else branch fallback
content = content.replace(
    "param = params_dict[chkpt_weight_name]",
    "mapped_name = _ct_map_name(chkpt_weight_name)\n            param = params_dict.get(chkpt_weight_name) or params_dict.get(mapped_name)",
)

# Remove the old CT fallback that was added earlier (cleanup)
content = content.replace(
    "param = params_dict.get(chkpt_weight_name)\n            if param is None:\n                # CT fallback",
    "param = params_dict.get(chkpt_weight_name) or params_dict.get(mapped_name)\n            if param is None:\n                # CT fallback",
)

open(path, "w").write(content)
print("ADDED _ct_map_name helper and applied to all param lookups")
