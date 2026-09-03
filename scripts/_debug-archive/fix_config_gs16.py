"""Fix config.json group_size for gs16 model."""

import json

CFG = r"C:\Users\ttimm\Documents\Project Portfolio\zaya1-godspeed\zaya1-8b-nvfp4-ct-gs16\config.json"
with open(CFG) as f:
    cfg = json.load(f)
qc = cfg["quantization_config"]
for gn, gc in qc["config_groups"].items():
    gc["weights"]["group_size"] = 16
with open(CFG, "w") as f:
    json.dump(cfg, f, indent=2)
print("Updated config group_size to 16")
with open(CFG) as f:
    cfg2 = json.load(f)
for gn, gc in cfg2["quantization_config"]["config_groups"].items():
    print(f"  {gn}: group_size={gc['weights']['group_size']}")
