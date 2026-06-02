"""Quick quality test: verify NVFP4 CT model integrity and decompress correctness."""

from __future__ import annotations

import logging
import os
import sys
from collections import Counter

import safetensors.torch as st
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("qcheck")

CT_PATH = "zaya1-8b-nvfp4-ct/model.safetensors"


def main() -> int:
    log.info("=== Model Integrity ===")
    state = st.load_file(CT_PATH, device="cpu")
    log.info("Tensors: %d | File: %.2f GB", len(state), os.path.getsize(CT_PATH) / 1e9)

    dc: Counter[str] = Counter()
    for t in state.values():
        dc[str(t.dtype)[6:]] += 1
    for dt in sorted(dc):
        log.info("  %s: %d", dt, dc[dt])

    # Verify no zero points
    zp = [n for n in state if "zero_point" in n]
    assert len(zp) == 0, f"{len(zp)} zero points found — symmetric NVFP4 should have 0"
    log.info("Zero points: 0 (correct for symmetric)")

    # Verify packing shape
    checks = [
        "lm_head.weight_packed",
        "model.layers.0.self_attn.o_proj.weight_packed",
        "model.layers.1.zaya_block.experts.local_experts.0.linear_fc1.weight_packed",
    ]
    for name in checks:
        if name not in state:
            continue
        t = state[name]
        s = state.get(name.replace("weight_packed", "weight_scale"))
        log.info("%s: shape=%s dtype=%s", name, list(t.shape), t.dtype)
        if s is not None:
            # Scale shape = [out, in//16] — computed on original in_features, not packed
            expected_cols = t.shape[1] * 2 // 16  # packed has in//2, scale has in//16
            assert s.shape[1] == expected_cols, f"Scale mismatch: {s.shape[1]} vs {expected_cols}"
            log.info("  scale: shape=%s dtype=%s (group_size=16 OK)", list(s.shape), s.dtype)

    # Decompress lm_head using NVFP4PackedCompressor (proper API)
    from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
    from compressed_tensors.quantization import preset_name_to_scheme

    d = {}
    for k in ("weight_packed", "weight_scale", "weight_global_scale"):
        key = f"lm_head.{k}"
        if key in state:
            d[k] = state[key].to("cuda:0")

    scheme = preset_name_to_scheme("NVFP4A16", targets=["Linear"])
    comp_state = {k: d[k] for k in d}
    decomp = NVFP4PackedCompressor.decompress(comp_state, scheme)
    deq = decomp["weight"]
    log.info(
        "Packed: %s → Deq: %s (via NVFP4PackedCompressor.decompress)", list(d["weight_packed"].shape), list(deq.shape)
    )

    from transformers import AutoModelForCausalLM

    log.info("Loading original BF16...")
    orig = AutoModelForCausalLM.from_pretrained(
        "Zyphra/ZAYA1-8B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    orig_w = orig.lm_head.weight.data
    error = (deq.to("cpu") - orig_w).abs().mean().item()
    rel_err = error / orig_w.abs().mean().item() * 100
    log.info("Original mean_abs: %.4f", orig_w.abs().mean().item())
    log.info("Absolute error: %.4f | Relative: %.1f%%", error, rel_err)
    log.info("(Expected 1-3%% for 4-bit quantization)")

    assert rel_err < 10.0, f"Relative error {rel_err:.1f}% exceeds 10% threshold!"
    del orig, state, d
    torch.cuda.empty_cache()
    log.info("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
