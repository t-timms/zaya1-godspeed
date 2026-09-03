#!/usr/bin/env python3
"""Move NVFP4 MoE check before WNA16 — NVFP4 takes priority for FP4 weights."""

p = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py"
c = open(p).read()

# Find the WNA16 if block start and the NVFP4 elif
old = c

# Strategy: insert NVFP4 check BEFORE the WNA16 check
insert_point = "        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):"
nvfp4_check = """        if quant_config._is_nvfp4_format(weight_quant):
            from .compressed_tensors_moe_w4a4_nvfp4 import (
                CompressedTensorsW4A4Nvfp4MoEMethod,
            )
            _is_valid_nvfp4_activations = (
                quant_config._is_nvfp4_format(input_quant) or input_quant is None
            )
            if not _is_valid_nvfp4_activations:
                raise ValueError(
                    "For NVFP4 weights, input quantization must also be NVFP4 format ",
                    f"or None for NVFP4A16, found {input_quant}",
                )
            return CompressedTensorsW4A4Nvfp4MoEMethod(
                layer.moe_config, layer_name, use_a16=(input_quant is None)
            )

"""

if "NVFP4 priority" not in c:
    c = c.replace(insert_point, nvfp4_check + insert_point)
    # Remove the original elif block (it's now redundant since we check first)
    c = c.replace(
        '        elif quant_config._is_nvfp4_format(weight_quant):\n            from .compressed_tensors_moe_w4a4_nvfp4 import (\n                CompressedTensorsW4A4Nvfp4MoEMethod,\n            )\n\n            _is_valid_nvfp4_activations = (\n                quant_config._is_nvfp4_format(input_quant) or input_quant is None\n            )\n            if not _is_valid_nvfp4_activations:\n                raise ValueError(\n                    "For NVFP4 weights, input quantization must also be NVFP4 format ",\n                    f"or None for NVFP4A16, found {input_quant}",\n                )\n            return CompressedTensorsW4A4Nvfp4MoEMethod(\n                layer.moe_config, layer_name, use_a16=(input_quant is None)\n            )',
        "",
    )
    open(p, "w").write(c)
    print("NVFP4 MoE check moved before WNA16 — NVFP4 takes priority")
else:
    print("Already patched")
