import os
import sys

sys.path.insert(0, "/home/ttimm/vllm-src")
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"


def main():
    from vllm import LLM, SamplingParams

    model_path = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct-gs16"
    print("Loading model with VLLM_CUTLASS MoE backend...")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=256,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        moe_backend="vllm_cutlass",
    )
    print("Running inference...")
    sp = SamplingParams(temperature=0, max_tokens=40)
    outputs = llm.generate(["The capital of France is"], sp)
    for o in outputs:
        text = o.outputs[0].text
        tokens = o.outputs[0].token_ids
        print("Prompt:", "The capital of France is")
        print("Tokens:", tokens[:10], "...")
        print("Text:", repr(text))
    print("DONE")


if __name__ == "__main__":
    main()
