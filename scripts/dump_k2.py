import json

p = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct/model.safetensors.index.json"
d = json.load(open(p))
for k in sorted(d["weight_map"].keys()):
    if "expert" in k.lower() or "w13" in k or "w2_weight" in k or "linear_fc" in k:
        print(k[:120])
    if "local_experts" in k:
        print("LOCAL_EXPERTS:", k)
