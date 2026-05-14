from vllm.model_executor.models.registry import ModelRegistry

print("Zaya registered:", "ZayaForCausalLM" in str(ModelRegistry.get_supported_archs()))

try:
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_nvfp4 import CompressedTensorsW4A16Fp4
    print("CT W4A16Fp4: OK")
except Exception as e:
    print(f"CT W4A16Fp4: MISSING - {e}")

try:
    from vllm.model_executor.layers.quantization.compressed_tensors import CompressedTensorsConfig
    print("CompressedTensorsConfig: OK")
except Exception as e:
    print(f"CompressedTensorsConfig: MISSING - {e}")
