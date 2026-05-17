"""Check if ZayaForCausalLM is registered."""
with open("/home/ttimm/vllm-src/vllm/model_executor/models/registry.py") as f:
    lines = f.readlines()

print("Lines containing Zaya or Zamba:")
for i, line in enumerate(lines, 1):
    if "Zaya" in line or "Zamba" in line:
        print(f"  Line {i}: {line.rstrip()}")

print()
if any("Zaya" in l for l in lines):
    print("ZayaForCausalLM IS registered")
else:
    print("ZayaForCausalLM is NOT registered")
