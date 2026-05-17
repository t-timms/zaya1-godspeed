"""Fix f-string bugs in zaya.py."""
path = "/home/ttimm/vllm-src/vllm/model_executor/models/zaya.py"

with open(path) as f:
    content = f.read()

# Fix 1: f-string missing 'f' prefix
old1 = '"WARNING: key {chkpt_weight_name} not in params! Skipping loading"'
new1 = 'f"WARNING: key {chkpt_weight_name} not in params! Skipping loading"'

if old1 in content:
    content = content.replace(old1, new1)
    print("Fixed: f-string format bug (added f prefix)")
else:
    print("f-string already fixed or pattern not found")
    # Check what's there now
    for i, line in enumerate(content.splitlines(), 1):
        if "chkpt_weight_name" in line and "WARNING" in line:
            print(f"  Line {i}: {line.strip()}")

# Fix 2: "fWARNING" (from broken sed)
if '"fWARNING:' in content:
    content = content.replace('"fWARNING:', 'f"WARNING:')
    print("Fixed: 'fWARNING' -> 'f\"WARNING' (sed artifact)")

with open(path, "w") as f:
    f.write(content)
print("Done.")
