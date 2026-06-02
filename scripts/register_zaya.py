"""Register ZayaForCausalLM in vLLM ModelRegistry."""

import sys

registry_path = "/home/ttimm/vllm-src/vllm/model_executor/models/registry.py"

with open(registry_path) as f:
    content = f.read()

if "ZayaForCausalLM" in content:
    print("ZayaForCausalLM already registered")
    sys.exit(0)

old = '    "Zamba2ForCausalLM": ("zamba2", "Zamba2ForCausalLM"),'
new = '    "ZayaForCausalLM": ("zaya", "ZayaForCausalLM"),\n    "Zamba2ForCausalLM": ("zamba2", "Zamba2ForCausalLM"),'

if old in content:
    content = content.replace(old, new, 1)
    with open(registry_path, "w") as f:
        f.write(content)
    print("ZayaForCausalLM registered successfully")
else:
    print("ERROR: Could not find Zamba2ForCausalLM line")
    # Search for nearby lines
    for i, line in enumerate(content.splitlines(), 1):
        if "Zamba" in line or "Xverse" in line:
            print(f"  Line {i}: {line.strip()}")
