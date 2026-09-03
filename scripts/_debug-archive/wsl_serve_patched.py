#!/usr/bin/env python3
"""Monkey-patch MambaStateShapeCalculator then start vLLM serve."""

import sys

# 1. Monkey-patch BEFORE any vLLM import
import vllm.model_executor.layers.mamba.mamba_utils as mu


def _cca_state_shape(cls, tp_world_size, conv_kernel_size, num_k_heads, num_q_heads, head_dim, hidden_size):
    proj_size = num_q_heads * head_dim
    proj_k_size = num_k_heads * head_dim
    return (
        cls._orient_conv_shape(proj_size // tp_world_size, conv_kernel_size - 1),
        cls._orient_conv_shape(proj_k_size // tp_world_size, conv_kernel_size - 1),
        (hidden_size,),
    )


def _cca_state_dtype(cls, model_dtype, mamba_cache_dtype):
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

    state_dtype = MambaStateDtypeCalculator.kda_state_dtype(model_dtype, mamba_cache_dtype)[0]
    return (state_dtype, state_dtype)


mu.MambaStateShapeCalculator.cca_state_shape = classmethod(_cca_state_shape)
mu.MambaStateDtypeCalculator.cca_state_dtype = classmethod(_cca_state_dtype)

print("Monkey-patched: cca_state_shape + cca_state_dtype")

if __name__ == "__main__":
    # 2. Import vLLM CLI
    from vllm.entrypoints.cli.main import main

    sys.argv = [
        "vllm",
        "serve",
        "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct",
        "--port",
        "8020",
        "--dtype",
        "float16",
        "--max-model-len",
        "2048",
        "--trust-remote-code",
        "--enforce-eager",
        "--max-num-seqs",
        "1",
        "--tokenizer",
        "Zyphra/ZAYA1-8B",
    ]
    main()
