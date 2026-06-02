import sys

sys.path.insert(0, "/home/ttimm/vllm-src")


def main():
    from vllm import LLM, SamplingParams

    model_path = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct-gs16"
    print("Loading model...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=256,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    print("Running inference...")
    sp = SamplingParams(temperature=0, max_tokens=40)
    prompts = [
        "The capital of France is",
        "Explain what a binary search tree is in one sentence.",
    ]
    outputs = llm.generate(prompts, sp)
    for prompt, output in zip(prompts, outputs):
        text = output.outputs[0].text
        tokens = output.outputs[0].token_ids
        print("Prompt:", prompt)
        print("Tokens:", tokens[:10], "...")
        print("Text:", repr(text))
        print()
    print("DONE")


if __name__ == "__main__":
    main()
