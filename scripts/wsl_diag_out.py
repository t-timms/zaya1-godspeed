#!/usr/bin/env python3
"""Add output diagnostic to MoE apply — check if loop produces anything."""
from pathlib import Path
p = Path("/home/ttimm/vllm-src/vllm/model_executor/layers/quantization/"
         "compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py")
c = p.read_text()

# Add diagnostic after the per-expert loop, before return
old_return = "        return out"
if old_return in c:
    new_return = """        if not hasattr(self, "_diag_loop_done"):
            self._diag_loop_done = True
            import logging
            _lg = logging.getLogger(__name__)
            _lg.warning(
                "MoE loop done: topk_ids unique=%s topk_weights range=[%.4f,%.4f] out mean=%.6f min=%.6f max=%.6f nonzero=%d/%d",
                sorted(topk_ids.unique().tolist()),
                topk_weights.min().item(), topk_weights.max().item(),
                out.float().mean().item(), out.float().min().item(), out.float().max().item(),
                int((out != 0).sum().item()), out.numel(),
            )
        return out"""
    c = c.replace(old_return, new_return)
    p.write_text(c)
    print("Output diagnostic added")
else:
    print("Pattern not found")
