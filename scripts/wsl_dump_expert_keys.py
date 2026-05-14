import json

idx = json.load(
    open("/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct/model.safetensors.index.json")
)
keys = list(idx["weight_map"].keys())
# Find keys with "experts"
for k in keys:
    if "experts" in k and ("w13" in k or "w2_weight" in k or "gate" in k):
        print(k)
