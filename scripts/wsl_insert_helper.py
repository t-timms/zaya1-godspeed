#!/usr/bin/env python3
"""Insert _is_small_dim_linear helper before CompressedTensorsConfig class."""

path = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
content = open(path).read()

helper = '''
def _is_small_dim_linear(layer):
    """Check if Linear layer output dim is too small for any WNA16 kernel."""
    from vllm.model_executor.layers.linear import LinearBase
    if not hasattr(layer, "output_size_per_partition"):
        return False
    return layer.output_size_per_partition < 64
'''

if "def _is_small_dim_linear" not in content:
    content = content.replace("class CompressedTensorsConfig", helper + "class CompressedTensorsConfig")
    open(path, "w").write(content)
    print("INSERTED _is_small_dim_linear")
else:
    print("ALREADY PRESENT")
