"""Direct test of vllm's CutlassExpertsFp4 MoE Group-MM kernel on SM120.

Adapted from /home/ttimm/vllm-src/tests/kernels/moe/test_nvfp4_moe.py with
the SM100 module-level skip bypassed. Uses Zaya-1's MoE shape (E=16, topk=1,
hidden=2048, intermediate=2048) to test the exact configuration that
collapses to NaN in the smoke-test.

If this passes (atol/rtol ~0.1) the kernel itself is correct on SM120
and the bug is in CutlassExpertsFp4's scale/weight fusion for our
checkpoint. If it fails, the kernel itself needs a fix or fallback.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/ttimm/vllm-src")
sys.path.insert(0, "/home/ttimm/vllm-src/tests")
sys.path.insert(0, "/home/ttimm/vllm-src/tests/kernels/quantization")

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from kernels.moe.utils import make_dummy_moe_config, make_test_weights  # noqa: E402
from kernels.utils import torch_moe  # noqa: E402
from nvfp4_utils import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, dequantize_nvfp4_to_dtype  # noqa: E402
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import CutlassExpertsFp4
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager  # noqa: E402


def run_one(
    m: int, n: int, k: int, e: int, topk: int, dtype: torch.dtype,
    a_scale: float = 1.0, realistic_gs: bool = False,
) -> tuple[float, float, bool]:
    set_random_seed(7)
    quant_blocksize = 16

    a = torch.randn((m, k), device="cuda", dtype=dtype) * a_scale

    (_, w1_q, w1_blockscale, w1_gs), (_, w2_q, w2_blockscale, w2_gs) = make_test_weights(
        e, n, k, in_dtype=dtype, quant_dtype="nvfp4", block_shape=None, per_out_ch_quant=False,
    )

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

    if realistic_gs:
        # Match what Zaya checkpoint loads: a_gscale = 2688/max_abs(a) per expert.
        max_abs_a = a.abs().max().item()
        a1_gs = torch.full((e,), (448.0 * 6.0) / max(max_abs_a, 1e-6), device="cuda", dtype=torch.float32)
        a2_gs = torch.full((e,), (448.0 * 6.0) / max(max_abs_a, 1e-6), device="cuda", dtype=torch.float32)
    else:
        a1_gs = torch.ones((e,), device="cuda", dtype=torch.float32)
        a2_gs = torch.ones((e,), device="cuda", dtype=torch.float32)

    quant_config = nvfp4_moe_quant_config(
        g1_alphas=(1 / w1_gs),
        g2_alphas=(1 / w2_gs),
        a1_gscale=a1_gs,
        a2_gscale=a2_gs,
        w1_scale=w1_blockscale,
        w2_scale=w2_blockscale,
    )
    moe_config = make_dummy_moe_config()
    kernel = mk.FusedMoEKernel(
        maybe_make_prepare_finalize(
            moe=moe_config, quant_config=quant_config, allow_new_interface=True, use_monolithic=False,
        ),
        CutlassExpertsFp4(moe_config=moe_config, quant_config=quant_config),
        inplace=False,
    )

    cutlass_output = kernel.apply(
        hidden_states=a, w1=w1_q, w2=w2_q,
        topk_weights=topk_weights, topk_ids=topk_ids,
        global_num_experts=e, activation=mk.MoEActivation.SILU,
        apply_router_weight_on_input=False, expert_map=None,
    )

    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a.flatten(), dim=-1)
    ).to(torch.float32)
    a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4, a_scale_interleaved, a_global_scale,
        dtype=a.dtype, device=a.device, block_size=quant_blocksize,
    )

    w1_d = torch.empty((e, 2 * n, k), device="cuda", dtype=dtype)
    w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)
    for idx in range(e):
        w1_d[idx] = dequantize_nvfp4_to_dtype(
            w1_q[idx], w1_blockscale[idx], w1_gs[idx],
            dtype=dtype, device=w1_q.device, block_size=quant_blocksize,
        )
        w2_d[idx] = dequantize_nvfp4_to_dtype(
            w2_q[idx], w2_blockscale[idx], w2_gs[idx],
            dtype=dtype, device=w2_q.device, block_size=quant_blocksize,
        )

    torch_output = torch_moe(a_in_dtype, w1_d, w2_d, score, topk)

    has_nan = torch.isnan(cutlass_output).any().item()
    abs_err = (cutlass_output.float() - torch_output.float()).abs()
    mae = abs_err.max().item()
    mre = (abs_err / torch_output.float().abs().clamp(min=1e-6)).mean().item()
    return mae, mre, has_nan


def main() -> int:
    cc = torch.cuda.get_device_capability(0)
    dev = torch.cuda.get_device_name(0)
    print(f"GPU: {dev}  compute_capability={cc[0]}.{cc[1]}")
    init_workspace_manager(torch.device("cuda:0"))
    with set_current_vllm_config(VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))):
        # First column = activation magnitude scale, second = whether to use
        # realistic a_gscale (2688/max_abs(a)) like the Zaya checkpoint loads.
        cases = [
            (8, 2048, 2048, 16, 1, 0.1, False),  # vanilla synthetic (control)
            (8, 2048, 2048, 16, 1, 1.0, False),  # larger inputs, a_gs=1
            (8, 2048, 2048, 16, 1, 1.0, True),   # realistic a_gscale
            (8, 2048, 2048, 16, 1, 10.0, True),  # realistic + Zaya-magnitude inputs
        ]
        print()
        print(f"{'m':>4} {'n':>5} {'k':>5} {'e':>3} {'topk':>5} {'a_sc':>6} {'gs?':>4} {'mae':>10} {'mre':>10}  nan?")
        fails = 0
        for m, n, k, e, topk, a_sc, gs in cases:
            try:
                mae, mre, has_nan = run_one(
                    m, n, k, e, topk, torch.bfloat16,
                    a_scale=a_sc, realistic_gs=gs,
                )
                gs_lbl = "Y" if gs else "N"
                nan_lbl = "YES" if has_nan else "no"
                print(
                    f"{m:>4} {n:>5} {k:>5} {e:>3} {topk:>5} {a_sc:>6.1f} "
                    f"{gs_lbl:>4} {mae:>10.4f} {mre:>10.4f}  {nan_lbl:>4}"
                )
                if has_nan:
                    fails += 1
            except Exception as ex:
                print(f"{m:>4} {n:>5} {k:>5} {e:>3} {topk:>5}  EXC: {type(ex).__name__}: {ex}")
                fails += 1
        print()
        print(f"FAILS: {fails}")
        return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
