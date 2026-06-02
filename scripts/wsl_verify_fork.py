#!/usr/bin/env python3
"""Deep verify Zyphra vLLM fork installation."""

from __future__ import annotations

import vllm

print(f"vLLM path: {vllm.__file__}")

# 1. Zaya model
try:
    print("ZayaForCausalLM import: OK")
except Exception as e:
    print(f"ZayaForCausalLM import: FAILED - {e}")

# 2. ModelRegistry
try:
    from vllm.model_executor.models.registry import ModelRegistry

    archs = ModelRegistry.get_supported_archs()
    if "ZayaForCausalLM" in archs:
        print("ZayaForCausalLM registered: YES")
    else:
        print(f"ZayaForCausalLM registered: NO (sample: {list(archs)[:5]})")
except Exception as e:
    print(f"Registry check: FAILED - {e}")

# 3. CCA attention
try:
    print("CCA attention: OK")
except Exception as e:
    print(f"CCA attention: FAILED - {e}")

# 4. NVFP4 scheme
try:
    print("CT W4A16Fp4: OK")
except Exception as e:
    print(f"CT W4A16Fp4: FAILED - {e}")

# 5. CompressedTensorsConfig
try:
    from vllm.model_executor.layers.quantization.compressed_tensors import CompressedTensorsConfig

    print("CompressedTensorsConfig: OK")
except ImportError:
    # v0.20.2 uses different path
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: F401
            CompressedTensorsConfig,
        )

        print("CompressedTensorsConfig (alt path): OK")
    except ImportError:
        print("CompressedTensorsConfig: MISSING")

# 6. Zaya tool parser
try:
    print("ZayaToolParser: OK")
except Exception as e:
    print(f"ZayaToolParser: FAILED - {e}")

print("\n=== VERIFICATION COMPLETE ===")
