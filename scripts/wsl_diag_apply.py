#!/usr/bin/env python3
"""Add one-shot diagnostic to MoE apply for debugging all-zero output."""
from pathlib import Path

p = Path("/home/ttimm/vllm-src/vllm/model_executor/layers/quantization/"
         "compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py")
c = p.read_text()

# Add diagnostic after dequant calls in apply()
old = """        w13_fp = self._dequant_experts(
            layer.w13_weight, self._w13_scale, out_dtype=x.dtype
        )
        w2_fp = self._dequant_experts(
            layer.w2_weight, self._w2_scale, out_dtype=x.dtype
        )"""

new = """        w13_fp = self._dequant_experts(
            layer.w13_weight, self._w13_scale, out_dtype=x.dtype
        )
        w2_fp = self._dequant_experts(
            layer.w2_weight, self._w2_scale, out_dtype=x.dtype
        )
        if not hasattr(self, "_diag_done"):
            self._diag_done = True
            import logging
            _lg = logging.getLogger(__name__)
            _lg.warning(
                "MoE diag x=%s w13=%s w2=%s ids=%s wts=%s",
                list(x.shape), list(w13_fp.shape), list(w2_fp.shape),
                list(topk_ids.shape), list(topk_weights.shape),
            )
            _lg.warning(
                "MoE diag w13 mean=%.6f min=%.6f max=%.6f",
                w13_fp.float().mean().item(),
                w13_fp.float().min().item(),
                w13_fp.float().max().item(),
            )
            _lg.warning(
                "MoE diag w2 mean=%.6f min=%.6f max=%.6f",
                w2_fp.float().mean().item(),
                w2_fp.float().min().item(),
                w2_fp.float().max().item(),
            )"""

if old in c and "MoE diag" not in c:
    c = c.replace(old, new)
    p.write_text(c)
    print("Diagnostic added to apply()")
else:
    print("Already has diagnostic or pattern not found")
