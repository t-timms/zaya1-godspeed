"""Round-trip verification: dequantize a few Linears from the patched W4A4
checkpoint and compare against the original BF16 weights from Zyphra/ZAYA1-8B.

If our global-scale convention is correct, max-abs error should be ≈
max_abs(W_orig) / 16 (one fp4 group quantization step) and mean error in
the 0.5-2% range. If convention is wrong, errors will be massive
(orders of magnitude off).
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/ttimm/vllm-src")

import safetensors.torch as st
import torch
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8

CKPT = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-w4a4/model.safetensors"


def dequantize_linear(state: dict, prefix: str) -> torch.Tensor:
    packed = state[f"{prefix}.weight_packed"]  # uint8, [out, in//2]
    scale_fp8 = state[f"{prefix}.weight_scale"]  # fp8, [out, in//16]
    wgs = state[f"{prefix}.weight_global_scale"].float()  # fp32 scalar/(1,)

    out_dim, packed_in = packed.shape
    in_dim = packed_in * 2

    unpacked = unpack_fp4_from_uint8(packed, out_dim, in_dim).float()  # [out, in] in fp4

    scale_f32 = scale_fp8.float()  # [out, in//16]
    if wgs.numel() == 1:
        wgs = wgs.item()
    effective_scale = scale_f32 / wgs  # [out, in//16] — recovers s_true

    group_size = 16
    n_groups = in_dim // group_size
    unpacked_groups = unpacked.view(out_dim, n_groups, group_size)
    scale_groups = effective_scale.view(out_dim, n_groups, 1)
    dequant_groups = unpacked_groups * scale_groups
    return dequant_groups.view(out_dim, in_dim)


def main() -> int:
    print(f"Loading patched checkpoint {CKPT} ...")
    state = st.load_file(CKPT, device="cpu")
    print(f"  loaded {len(state)} tensors")

    print("\nLoading original ZAYA1-8B BF16 weights ...")
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(
        "Zyphra/ZAYA1-8B",
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    orig_sd = dict(model.named_parameters())
    del model
    print(f"  loaded {len(orig_sd)} params")

    probe_prefixes = [
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.zaya_block.experts.local_experts.0.linear_fc1",
        "model.layers.0.zaya_block.experts.local_experts.0.linear_fc2",
        "model.layers.40.self_attn.o_proj",
    ]

    print(f"\n{'prefix':<70} {'max_abs_orig':>12} {'max_abs_err':>12} {'mean_abs_err':>12} {'mae_rel':>10}")
    for prefix in probe_prefixes:
        if f"{prefix}.weight_packed" not in state:
            print(f"{prefix}: skip (not in checkpoint)")
            continue
        orig_key = f"{prefix}.weight"
        if orig_key not in orig_sd:
            print(f"{prefix}: skip (orig BF16 weight not found)")
            continue

        w_orig = orig_sd[orig_key].float()
        w_dequant = dequantize_linear(state, prefix)

        if w_dequant.shape != w_orig.shape:
            print(f"{prefix}: SHAPE MISMATCH dequant={tuple(w_dequant.shape)} orig={tuple(w_orig.shape)}")
            continue

        max_abs_orig = w_orig.abs().max().item()
        err = (w_dequant - w_orig).abs()
        max_abs_err = err.max().item()
        mean_abs_err = err.mean().item()
        mae_rel = mean_abs_err / max_abs_orig if max_abs_orig > 0 else float("nan")
        print(f"{prefix:<70} {max_abs_orig:>12.4f} {max_abs_err:>12.4f} {mean_abs_err:>12.4f} {mae_rel:>10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
