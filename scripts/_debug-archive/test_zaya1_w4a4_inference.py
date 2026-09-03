"""Week 2 step 1 smoke-test: load W4A4 NVFP4 checkpoint via vllm-src and
generate one short completion. Forces the CUTLASS SM120 backend and disables
Marlin so we know any failure is on the W4A4 path, not a Marlin fallback.

Run from WSL:
    source /home/ttimm/vllm-env/bin/activate
    python3 /mnt/c/Users/ttimm/Documents/Project\\ Portfolio/zaya1-godspeed/scripts/test_zaya1_w4a4_inference.py
"""

from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("VLLM_NVFP4_GEMM_BACKEND", "cutlass")
os.environ.setdefault("VLLM_DISABLED_KERNELS", "Marlin")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

sys.path.insert(0, "/home/ttimm/vllm-src")

MODEL_PATH = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-w4a4"


def main() -> int:
    from vllm import LLM, SamplingParams

    print(f"VLLM_NVFP4_GEMM_BACKEND={os.environ['VLLM_NVFP4_GEMM_BACKEND']}")
    print(f"VLLM_DISABLED_KERNELS={os.environ['VLLM_DISABLED_KERNELS']}")
    print(f"Loading {MODEL_PATH} ...")

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=128,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        max_num_seqs=1,
        moe_backend="cutlass",  # VLLM_CUTLASS (compiled SM120 kernels)
    )
    print("Model loaded. Running inference...")

    sampling_params = SamplingParams(temperature=0, max_tokens=32, logprobs=5)
    prompt = "The capital of France is"
    outputs = llm.generate([prompt], sampling_params)
    out = outputs[0].outputs[0]
    print(f"Prompt: {prompt!r}")
    print(f"Output text:      {out.text!r}")
    print(f"Output token_ids: {list(out.token_ids)}")
    print(f"Finish reason:    {out.finish_reason}")
    print("--- top-5 logprobs at each step ---")
    if out.logprobs is not None:
        for i, step in enumerate(out.logprobs):
            entries = sorted(step.items(), key=lambda kv: -kv[1].logprob)[:5]
            print(f"step {i}:")
            for tok_id, lp in entries:
                print(f"  tok={tok_id:>6}  logprob={lp.logprob:>10.4f}  decoded={lp.decoded_token!r}")
    else:
        print("(no logprobs available)")
    if "Paris" in out.text:
        print("SMOKE TEST PASSED — coherent output, CUTLASS path live")
        return 0
    print("SMOKE TEST INCOHERENT — output non-Paris but load succeeded")
    return 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 2
    sys.exit(rc)
