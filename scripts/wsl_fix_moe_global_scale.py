#!/usr/bin/env python3
"""Fix: Initialize NVFP4 MoE global/input scales to 1.0 in create_weights.

Bug: create_weights uses torch.empty() for weight_global_scale and input_scale.
The CT checkpoint doesn't store these (symmetric per-group quantization).
Uninitialized values → garbage Marlin kernel outputs → all pad tokens.

Fix: torch.empty → torch.ones in create_weights. Checkpoint loading overwrites
if data exists; otherwise defaults to 1.0 (identity, no per-channel rescaling).
"""

from pathlib import Path

MOE_PATH = Path(
    "/home/ttimm/vllm-env/lib/python3.12/site-packages/"
    "vllm/model_executor/layers/quantization/compressed_tensors/"
    "compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py"
)


def fix() -> bool:
    content = MOE_PATH.read_text()
    modified = False

    replacements = [
        ("torch.empty(num_experts, w13_num_shards, dtype=torch.float32)",
         "torch.ones(num_experts, w13_num_shards, dtype=torch.float32)"),
        ("torch.empty(num_experts, dtype=torch.float32), requires_grad=False\n        )\n        layer.register_parameter(\"w2_weight_global_scale\"",
         "torch.ones(num_experts, dtype=torch.float32), requires_grad=False\n        )\n        layer.register_parameter(\"w2_weight_global_scale\""),
    ]

    # Input scales - need to find the right ones
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print("  [FIX] torch.empty → torch.ones")
            modified = True

    # Also fix input global scales
    for old in [
        "torch.empty(num_experts, w13_num_shards, dtype=torch.float32),\n            requires_grad=False,\n        )\n        layer.register_parameter(\"w13_input_global_scale\"",
        "torch.empty(num_experts, dtype=torch.float32), requires_grad=False\n        )\n        layer.register_parameter(\"w2_input_global_scale\"",
    ]:
        if old in content:
            content = content.replace(old, old.replace("torch.empty", "torch.ones"))
            print("  [FIX] torch.empty → torch.ones for input scale")
            modified = True

    # Remove old unreliable zero/NaN check from process_weights_after_loading
    marker = "# Default global/input scales to 1.0 if not loaded from checkpoint."
    if marker in content:
        # Find the start and end of the block
        start = content.index(marker)
        end_marker = "        # Use a single gscale for w13."
        end = content.index(end_marker, start)
        content = content[:start] + content[end:]
        print("  [CLEANUP] Removed old unreliable zero/NaN check")
        modified = True

    if modified:
        MOE_PATH.write_text(content)
        print("  [SAVED] compressed_tensors_moe_w4a4_nvfp4.py updated")
    return modified


if __name__ == "__main__":
    if not MOE_PATH.exists():
        print(f"ERROR: NVFP4 MoE file not found at {MOE_PATH}")
        exit(1)
    ok = fix()
    if ok:
        print("\nFix applied. Re-run smoke test to verify.")
    else:
        print("\nNo changes needed or already applied.")
