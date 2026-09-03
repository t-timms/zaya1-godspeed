#!/usr/bin/env python3
"""Patch _is_nvfp4_format to accept group_size=64 and strategy=group."""

path = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
content = open(path).read()

# Fix 1: Make _is_nvfp4_format accept group_size=64 too
old1 = "        is_group_size_16 = quant_args.group_size == 16"
new1 = "        is_group_size_valid = quant_args.group_size in (16, 64)"
content = content.replace(old1, new1)

# Fix 2: Accept strategy="group" as well (not just "tensor_group")
old2 = """        is_tensor_group_quant = (
            quant_args.strategy == QuantizationStrategy.TENSOR_GROUP.value
        )"""
new2 = """        is_tensor_group_quant = (
            quant_args.strategy in (
                QuantizationStrategy.TENSOR_GROUP.value,
                QuantizationStrategy.GROUP.value,
            )
        )"""
content = content.replace(old2, new2)

open(path, "w").write(content)
print("PATCHED _is_nvfp4_format")
