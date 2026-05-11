# ZAYA NVFP4 vLLM Plugin
# Registers ZayaForCausalLM at vLLM startup via VLLM_PLUGINS mechanism.
# Usage: VLLM_PLUGINS=zaya_plugin vllm serve ...

def register():
    from vllm.model_executor.models.registry import ModelRegistry
    if "ZayaForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "ZayaForCausalLM",
            "vllm.model_executor.models.zaya:ZayaForCausalLM",
        )
        print("[ZAYA plugin] Registered ZayaForCausalLM")
