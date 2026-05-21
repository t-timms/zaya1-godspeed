from __future__ import annotations

import json
import sys
from pathlib import Path

from lm_eval import simple_evaluate


def main() -> int:
    MODEL = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-w4a4"

    results = simple_evaluate(
        model="vllm",
        model_args=(
            f"pretrained={MODEL},"
            "dtype=bfloat16,"
            "moe_backend=cutlass,"
            "gpu_memory_utilization=0.85,"
            "max_model_len=4096,"
            "tensor_parallel_size=1,"
            "kv_cache_dtype=fp8"
        ),
        tasks=["hellaswag"],
        num_fewshot=0,
        batch_size="auto",
        device="cuda",
        limit=None,
        random_seed=42,
        numpy_random_seed=42,
    )

    res = results.get("results", {})
    print(json.dumps(res, indent=2))

    out = Path("/tmp/lmeval_w4a4_hellaswag.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"Full results saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
