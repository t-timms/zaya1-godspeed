"""Fix ZayaForCausalLM registration format."""

path = "/home/ttimm/vllm-src/vllm/model_executor/models/registry.py"

with open(path) as f:
    content = f.read()

# Fix the malformed line
old = "    ZayaForCausalLM: (zaya, ZayaForCausalLM),"
new = '    "ZayaForCausalLM": ("zaya", "ZayaForCausalLM"),'

if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("Fixed ZayaForCausalLM registration format")
else:
    # Check current state
    for i, line in enumerate(content.splitlines(), 1):
        if "Zaya" in line:
            print(f"Line {i}: {line}")
