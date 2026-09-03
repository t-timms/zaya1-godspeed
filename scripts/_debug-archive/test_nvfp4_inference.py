"""Direct NVFP4 ZAYA1-8B inference test via vLLM Python API.

Bypasses CLI multiprocessing issues by using vLLM's LLM class directly.
"""

from __future__ import annotations

import json
import os
import sys

gguf_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/zaya1-8b-nvfp4.gguf"

# Register ZayaForCausalLM
from vllm.model_executor.models.registry import ModelRegistry  # noqa: E402

ModelRegistry.register_model(
    "ZayaForCausalLM",
    "vllm.model_executor.models.zaya:ZayaForCausalLM",
)
print("[ZAYA] Registered ZayaForCausalLM")

# Register zaya in gguf
import gguf  # noqa: E402

if "zaya" not in gguf.MODEL_ARCH_NAMES.values():
    zid = max(gguf.MODEL_ARCH_NAMES.keys()) + 1
    gguf.MODEL_ARCH_NAMES[zid] = "zaya"
    setattr(gguf.MODEL_ARCH, "ZAYA", zid)
    print("[ZAYA] Registered zaya in gguf")

# Load name map
name_map = {}
map_path = gguf_path.replace(".gguf", ".name_map.json")
if os.path.exists(map_path):
    with open(map_path) as f:
        name_map = json.load(f)
    print(f"[ZAYA] Loaded name map: {len(name_map)} entries")

# Patch gguf loader inline
import vllm.model_executor.model_loader.gguf_loader as gl  # noqa: E402

orig_load_model = gl.GGUFModelLoader.load_model


def patched_load_model(self, vllm_config, model_config, prefix=""):
    self._zaya_name_map = name_map
    return orig_load_model(self, vllm_config, model_config, prefix)


gl.GGUFModelLoader.load_model = patched_load_model

# Also patch the mapping in _get_gguf_weights_map
orig_get_weights = gl.GGUFModelLoader._get_gguf_weights_map


def patched_get_weights(self, model_config):
    if hasattr(self, "_zaya_name_map") and self._zaya_name_map:
        # Inject name_map entries into the gguf->hf mapping BEFORE the original method
        pass
    return orig_get_weights(self, model_config)


gl.GGUFModelLoader._get_gguf_weights_map = patched_get_weights

from vllm import LLM, SamplingParams  # noqa: E402

print(f"[ZAYA] Loading NVFP4 model from {gguf_path}...")
llm = LLM(
    model=gguf_path,
    tokenizer="Zyphra/ZAYA1-8B",
    hf_config_path="Zyphra/ZAYA1-8B",
    dtype="float16",
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    enforce_eager=True,
)
print("[ZAYA] Model loaded!")

# Test inference
prompt = "def fibonacci(n):"
print(f"[ZAYA] Testing: {prompt}")
output = llm.generate(prompt, SamplingParams(temperature=0.6, max_tokens=100))
print(output[0].outputs[0].text)
print("[ZAYA] Done!")
