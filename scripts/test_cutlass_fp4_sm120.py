"""Direct test of vllm's cutlass_scaled_fp4_mm on RTX 5070 Ti (SM120).

Adapted from /home/ttimm/vllm-src/tests/kernels/quantization/test_nvfp4_scaled_mm.py
with the SM100 gate removed. Verifies that the compiled SM120 NVFP4 kernels
produce numerically-correct outputs against a BF16 reference matmul.

If this test passes (atol/rtol ~0.1), the kernel is correct and the W4A4 model
collapse is upstream: MoE wiring, weight layout, or model-architecture forward.

If this test fails, the kernel itself is suspect on SM120 and we need either
to fix it or fall back to a different backend.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/ttimm/vllm-src")
sys.path.insert(0, "/home/ttimm/vllm-src/tests/kernels/quantization")

import torch
from nvfp4_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, dequantize_nvfp4_to_dtype
from vllm import _custom_ops as ops


def run_one(m: int, n: int, packed_k: int, dtype: torch.dtype, seed: int = 42) -> tuple[float, float]:
    torch.manual_seed(seed)
    device = "cuda:0"
    k = packed_k * 2
    block_size = 16

    a_dtype = torch.randn((m, k), dtype=dtype, device=device)
    b_dtype = torch.randn((n, k), dtype=dtype, device=device)

    a_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten(), dim=-1)).to(torch.float32)
    b_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten(), dim=-1)).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)

    a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a_dtype, a_global_scale)
    b_fp4, b_scale_interleaved = ops.scaled_fp4_quant(b_dtype, b_global_scale)

    a_recon = dequantize_nvfp4_to_dtype(
        a_fp4, a_scale_interleaved, a_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    b_recon = dequantize_nvfp4_to_dtype(
        b_fp4, b_scale_interleaved, b_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    expected_out = torch.matmul(a_recon, b_recon.t())

    out = ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, a_scale_interleaved, b_scale_interleaved, alpha, dtype)

    abs_err = (out.float() - expected_out.float()).abs()
    rel_err = abs_err / expected_out.float().abs().clamp(min=1e-6)
    return abs_err.max().item(), rel_err.mean().item()


def main() -> int:
    cc = torch.cuda.get_device_capability(0)
    dev = torch.cuda.get_device_name(0)
    print(f"GPU: {dev}  compute_capability={cc[0]}.{cc[1]}")

    shapes = [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
    print()
    print(f"{'m':>5} {'n':>5} {'k':>5} {'dtype':>10} {'max_abs_err':>12} {'mean_rel_err':>14}  pass?")
    fails = 0
    for m, n, pk in shapes:
        for dtype in [torch.float16, torch.bfloat16]:
            mae, mre = run_one(m, n, pk, dtype)
            ok = mae < 1.0 and mre < 0.2
            dt = str(dtype).removeprefix("torch.")
            status = "PASS" if ok else "FAIL"
            print(
                f"{m:>5} {n:>5} {pk * 2:>5} {dt:>10} {mae:>12.4f} {mre:>14.4f}  {status}"
            )
            if not ok:
                fails += 1
    print()
    print(f"FAILS: {fails}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
