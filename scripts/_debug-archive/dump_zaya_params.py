"""Dump the actual parameter names registered by the vllm-src Zaya model
to see why CCA + router checkpoint keys are being skipped during load.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("VLLM_NVFP4_GEMM_BACKEND", "cutlass")

sys.path.insert(0, "/home/ttimm/vllm-src")

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    set_default_torch_dtype,
)

MODEL_PATH = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-w4a4"


def main() -> int:
    args = EngineArgs(
        model=MODEL_PATH,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=128,
        enforce_eager=True,
        max_num_seqs=1,
    )
    cfg: VllmConfig = args.create_engine_config()

    import torch

    with set_default_torch_dtype(torch.bfloat16):
        model = initialize_model(vllm_config=cfg)

    pnames = [n for n, _ in model.named_parameters()]
    print(f"total params: {len(pnames)}")

    targets = (
        "self_attn.qkv.linear_q.weight",
        "self_attn.qkv.linear_k.weight",
        "self_attn.qkv.val_proj1.weight",
        "self_attn.qkv.val_proj2.weight",
        "zaya_block.router.down_proj.weight",
        "zaya_block.router.router_mlp.0.weight",
        "zaya_block.router.router_mlp.2.weight",
        "zaya_block.router.router_mlp.4.weight",
    )

    for target in targets:
        hits = [p for p in pnames if p.endswith(target) and "layers.1." in p]
        if hits:
            print(f"FOUND       {target}: {hits[0]}")
        else:
            related = [p for p in pnames if "layers.1." in p and target.split(".")[-2] in p][:5]
            print(f"MISSING     {target}  | nearby={related}")

    print()
    print("--- ALL layer.1 param names (sample) ---")
    for p in sorted(pnames):
        if "layers.1." in p and ("qkv" in p or "router" in p or "cca" in p):
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
