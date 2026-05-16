"""Quick inference test with zaya1 NVFP4 model to verify CUTLASS SM120 kernels."""
import sys
sys.path.insert(0, "/home/ttimm/vllm-src")

from vllm import LLM, SamplingParams

model_path = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct-gs16"

print("Loading zaya1-8b-nvfp4-ct-gs16...")
llm = LLM(
    model=model_path,
    dtype="bfloat16",
    trust_remote_code=True,
    max_model_len=512,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
)
print("Model loaded. Running inference...")

sampling_params = SamplingParams(temperature=0, max_tokens=20)
prompts = [
    "The capital of France is",
    "The meaning of life is",
]
outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Output: {text}")
    print()

print("INFERENCE TEST PASSED")
