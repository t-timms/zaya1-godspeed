from vllm.model_executor.models.registry import ModelRegistry

print("Zaya registered:", "ZayaForCausalLM" in str(ModelRegistry.get_supported_archs()))

try:
    print("CT W4A16Fp4: OK")
except Exception as e:
    print(f"CT W4A16Fp4: MISSING - {e}")

try:
    print("CompressedTensorsConfig: OK")
except Exception as e:
    print(f"CompressedTensorsConfig: MISSING - {e}")
