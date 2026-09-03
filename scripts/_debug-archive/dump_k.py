import json

p = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct/model.safetensors.index.json"
d = json.load(open(p))
for k in d["weight_map"]:
    if "experts" in k and ("w13" in k or "w2" in k):
        print(k)
print("DONE")
