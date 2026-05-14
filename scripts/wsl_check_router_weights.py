#!/usr/bin/env python3
"""Check if router weights exist in safetensors and their format."""

import json

idx = json.load(
    open("/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct/model.safetensors.index.json")
)
router_keys = [k for k in idx["weight_map"].keys() if "router" in k.lower()]
print(f"Router weight keys: {len(router_keys)}")
for k in sorted(router_keys)[:10]:
    print(f"  {k}")
print(f"\nTotal weight keys: {len(idx['weight_map'])}")

# Check specific router Linear weights vs regular Linear weights
from safetensors import safe_open

with safe_open(
    "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct/model.safetensors", framework="pt"
) as f:
    checks = [
        # Router weights
        "model.layers.1.zaya_block.router.router_mlp.0.weight",
        "model.layers.1.zaya_block.router.router_mlp.2.weight",
        # Regular Linear weights (attention)
        "model.layers.1.self_attn.o_proj.weight",
        "model.layers.1.self_attn.qkv.linear_q.weight",
        # Expert weights
        "model.layers.1.zaya_block.experts.0.gate_proj.weight",
    ]
    for k in checks:
        try:
            t = f.get_tensor(k)
            print(f"{k}: dtype={t.dtype}, shape={list(t.shape)}")
        except Exception:
            print(f"{k}: NOT FOUND")

    # Also check total tensors by dtype
    print()
    idx = json.load(
        open(
            "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct/model.safetensors.index.json"
        )
    )
    dtypes = {}
    for k in idx["weight_map"]:
        t = f.get_tensor(k)
        dt = str(t.dtype)
        dtypes[dt] = dtypes.get(dt, 0) + 1
    print("Tensor dtypes:")
    for dt, count in sorted(dtypes.items()):
        print(f"  {dt}: {count}")
