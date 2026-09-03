#!/usr/bin/env python3
"""Fix zaya.py load_weights for CT vLLM-native weight naming.

CT-quantized safetensors use vLLM-native names like:
  model.layers.N.zaya_block.experts.w13_weight
But vLLM's module hierarchy has an extra 'mlp' level:
  model.layers.N.mlp.zaya_block.experts.w13_weight

This patch adds a fallback that tries inserting 'mlp.' into the key.
"""

path = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
content = open(path).read()

old_fallback = """        else:
            param = params_dict[chkpt_weight_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(chkpt_weight_name)"""

new_fallback = """        else:
            param = params_dict.get(chkpt_weight_name)
            if param is None:
                # CT fallback: vLLM-native names may omit 'mlp.' prefix.
                # e.g., 'zaya_block.experts.w13_weight' vs
                # 'mlp.zaya_block.experts.w13_weight'
                alt_name = chkpt_weight_name.replace(
                    ".zaya_block.", ".mlp.zaya_block."
                )
                param = params_dict.get(alt_name)
            if param is None:
                raise KeyError(
                    f"Parameter {chkpt_weight_name} not found "
                    f"(also tried {alt_name})"
                )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(chkpt_weight_name)"""

if "CT fallback" not in content:
    content = content.replace(old_fallback, new_fallback)
    open(path, "w").write(content)
    print("ADDED CT weight naming fallback to load_weights")
else:
    print("ALREADY PRESENT")
