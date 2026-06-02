"""Quick inference test with zaya1 NVFP4 model."""

import sys

sys.path.insert(0, "/home/ttimm/vllm-src")


def main():
    from vllm import LLM, SamplingParams

    model_path = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct-gs16"

    print("Loading zaya1-8b-nvfp4-ct-gs16...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=128,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_num_seqs=1,
    )
    print("Model loaded. Running inference...")

    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    prompt = "The capital of France is"
    outputs = llm.generate([prompt], sampling_params)
    text = outputs[0].outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Output: {text}")
    print()
    if "Paris" in text:
        print("INFERENCE TEST PASSED — coherent output confirmed")
    else:
        print(f"Output received: '{text}'")


if __name__ == "__main__":
    main()
