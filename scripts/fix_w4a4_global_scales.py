"""In-place patch for the W4A4 NVFP4 checkpoint to use compressed-tensors'
correct global-scale convention.

CT's NVFP4 contract (from compressed_tensors.quantization.utils.helpers
``_compute_global_scale``):

    global_scale = FP8_E4M3_MAX * FP4_E2M1_MAX / max_abs(tensor)
                 = 448 * 6 / max_abs
                 = 2688 / max_abs

and per-group ``weight_scale`` is stored as fp8 in the form ``s_true * global_scale``
so that ``effective_scale = s_fp8 / global_scale = s_true`` recovers the original
dequantization factor.

The original quantize_zaya_ct_nvfp4.py used the inverse convention
(``s_true = max_abs / 6`` and never set ``weight_global_scale``), which produces
uninitialized ``weight_global_scale`` values and a per-tensor scale mismatch
when fed to vLLM's CUTLASS path. This script patches the on-disk safetensors
without re-running calibration.

What it does, per Linear (1320 of them):
  1. Read existing fp8 ``weight_scale`` (currently = ``w_max_group / 6``).
  2. Compute ``max_abs(W) = 6 * max(weight_scale)``.
  3. Compute ``weight_global_scale = 2688 / max_abs(W)``.
  4. Rewrite ``weight_scale = clamp(weight_scale * weight_global_scale, max=448)``,
     cast back to fp8_e4m3fn.
  5. Compute ``input_global_scale_new = 448 / input_global_scale_old`` (the
     algebraic conversion from ``max_act/6`` to ``2688/max_act``).

Run from WSL (vllm-env activated, no GPU needed):
    python3 scripts/fix_w4a4_global_scales.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import safetensors.torch as st
import torch

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
GLOBAL_SCALE_NUM = FP8_E4M3_MAX * FP4_E2M1_MAX  # 2688.0

DEFAULT_PATH = Path("/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-w4a4")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _module_of(key: str) -> str:
    return key.rsplit(".", 1)[0]


def patch_checkpoint(model_dir: Path, in_place: bool = True) -> int:
    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        logger.error("missing %s", safetensors_path)
        return 1

    logger.info("Loading %s ...", safetensors_path)
    t0 = time.time()
    state = st.load_file(str(safetensors_path), device="cpu")
    logger.info("Loaded %d tensors in %.0fs", len(state), time.time() - t0)

    by_module: dict[str, dict[str, str]] = defaultdict(dict)
    for k in state:
        if k.endswith(
            (
                "weight_scale",
                "weight_global_scale",
                "weight_packed",
                "input_global_scale",
            )
        ):
            suffix = k.rsplit(".", 1)[1]
            by_module[_module_of(k)][suffix] = k

    quantized_modules = [m for m, parts in by_module.items() if "weight_packed" in parts]
    logger.info("Found %d quantized Linears", len(quantized_modules))

    patched_wgs = 0
    patched_ws = 0
    patched_igs = 0
    skipped_no_ws = 0

    for i, mod in enumerate(quantized_modules):
        parts = by_module[mod]
        ws_key = parts.get("weight_scale")
        if ws_key is None:
            skipped_no_ws += 1
            continue
        ws_fp8 = state[ws_key]
        ws_f32 = ws_fp8.float()

        w_max_group_max = ws_f32.abs().max().item()
        max_abs_w = FP4_E2M1_MAX * w_max_group_max
        if max_abs_w <= 0:
            logger.warning("zero weight scale in %s; skipping", mod)
            continue

        wgs = GLOBAL_SCALE_NUM / max_abs_w
        wgs_tensor = torch.tensor([wgs], dtype=torch.float32)

        new_ws_f32 = ws_f32 * wgs
        new_ws_f32 = new_ws_f32.clamp(max=FP8_E4M3_MAX, min=-FP8_E4M3_MAX)
        new_ws_fp8 = new_ws_f32.to(torch.float8_e4m3fn)

        state[ws_key] = new_ws_fp8
        patched_ws += 1

        wgs_key = parts.get("weight_global_scale", f"{mod}.weight_global_scale")
        state[wgs_key] = wgs_tensor
        patched_wgs += 1

        igs_key = parts.get("input_global_scale")
        if igs_key is not None:
            igs_old = state[igs_key].float()
            if igs_old.numel() == 1 and igs_old.item() > 0:
                igs_new = (FP8_E4M3_MAX / igs_old).to(torch.float32)
                state[igs_key] = igs_new
                patched_igs += 1

        if (i + 1) % 200 == 0:
            logger.info("  %d/%d Linears patched ...", i + 1, len(quantized_modules))

    logger.info(
        "Patched: weight_scale=%d weight_global_scale=%d input_global_scale=%d (skipped no-weight-scale: %d)",
        patched_ws,
        patched_wgs,
        patched_igs,
        skipped_no_ws,
    )

    out_path = safetensors_path if in_place else safetensors_path.with_suffix(".patched.safetensors")
    logger.info("Saving to %s ...", out_path)
    t0 = time.time()
    st.save_file(state, str(out_path))
    logger.info("Saved in %.0fs", time.time() - t0)

    sample_keys = [
        k
        for k in [
            "model.layers.0.self_attn.o_proj.weight_global_scale",
            "model.layers.0.self_attn.o_proj.input_global_scale",
            "model.layers.0.self_attn.o_proj.weight_scale",
        ]
        if k in state
    ]
    for k in sample_keys:
        v = state[k]
        sample = v.float().flatten()[:4].tolist()
        logger.info("verify  %s  shape=%s  dtype=%s  sample=%s", k, tuple(v.shape), v.dtype, sample)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="write to model.patched.safetensors instead of overwriting",
    )
    args = parser.parse_args()
    return patch_checkpoint(args.path, in_place=not args.copy)


if __name__ == "__main__":
    sys.exit(main())
