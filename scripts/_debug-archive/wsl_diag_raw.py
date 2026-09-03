#!/usr/bin/env python3
"""Add raw packed data diagnostic to MoE _dequant_experts."""

from pathlib import Path

p = Path(
    "/home/ttimm/vllm-src/vllm/model_executor/layers/quantization/"
    "compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py"
)
c = p.read_text()

old = """        from compressed_tensors.compressors.nvfp4.helpers import (
            unpack_fp4_from_uint8,
        )
        from compressed_tensors.quantization.lifecycle.forward import (
            dequantize,
        )
        E, M, Ph = packed.shape"""

new = """        from compressed_tensors.compressors.nvfp4.helpers import (
            unpack_fp4_from_uint8,
        )
        from compressed_tensors.quantization.lifecycle.forward import (
            dequantize,
        )
        import logging
        E, M, Ph = packed.shape
        _lg = logging.getLogger(__name__)
        _lg.warning(
            "MoE dequant raw: E=%d M=%d Ph=%d packed_dtype=%s scale_dtype=%s",
            E, M, Ph, str(packed.dtype), str(scales.dtype),
        )
        _lg.warning(
            "MoE dequant raw: packed[0] min=%d max=%d scale[0] min=%.6f max=%.6f",
            int(packed[0].min().item()), int(packed[0].max().item()),
            scales[0].float().min().item(), scales[0].float().max().item(),
        )"""

if old in c and "MoE dequant raw" not in c:
    c = c.replace(old, new)
    p.write_text(c)
    print("Raw data diagnostic added")
else:
    print("Already has diagnostic or pattern not found")
