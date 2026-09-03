"""Check actual group_size from tensor shapes."""
from __future__ import annotations

import json

import safetensors.torch as st

CT_PATH = r"C:\Users\ttimm\Documents\Project Portfolio\zaya1-godspeed\zaya1-8b-nvfp4-ct"

# Load state
state = st.load_file(f"{CT_PATH}/model.safetensors", device="cpu")

# Check attention layer
for k in sorted(state.keys()):
    if "o_proj.weight_scale" in k:
        w = state[k]
        pk = k.replace("weight_scale", "weight_packed")
        if pk in state:
            pw = state[pk]
            in_feat = pw.shape[1] * 2
            gs = in_feat // w.shape[1]
            print(f"ATTN: {k}")
            print(f"  scale: {list(w.shape)}, weight: {list(pw.shape)}")
            print(f"  in_features={in_feat}, group_size={gs}")
        break

# Check MoE layer
for k in sorted(state.keys()):
    if "linear_fc1.weight_scale" in k and "local_experts.0" in k:
        w = state[k]
        pk = k.replace("weight_scale", "weight_packed")
        if pk in state:
            pw = state[pk]
            in_feat = pw.shape[1] * 2
            gs = in_feat // w.shape[1]
            print(f"\nMoE: {k}")
            print(f"  scale: {list(w.shape)}, weight: {list(pw.shape)}")
            print(f"  in_features={in_feat}, group_size={gs}")
        break

# Report config
print("\nConfig group_size:")
with open(f"{CT_PATH}/config.json") as f:
    cfg = json.load(f)
    qc = cfg["quantization_config"]
    for gn, gc in qc["config_groups"].items():
        wc = gc.get("weights", {})
        print(f"  {gn}: group_size={wc.get('group_size')}, strategy={wc.get('strategy')}")
