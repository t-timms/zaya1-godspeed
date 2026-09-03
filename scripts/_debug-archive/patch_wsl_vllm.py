"""Apply 3 patches to stock vLLM 0.20.2 in WSL2 for Zaya support."""

import os

VLLM_PATH = "/root/vllm-env/lib/python3.12/site-packages/vllm"

patches = []

# 1. Register ZayaForCausalLM in ModelRegistry
registry_path = os.path.join(VLLM_PATH, "model_executor", "models", "registry.py")
patches.append(
    (
        "ModelRegistry registration",
        registry_path,
        [
            # Find the registration block and add ZayaForCausalLM
            (
                "_TEXT_GENERATION_MODELS.update({",
                '_TEXT_GENERATION_MODELS.update({\n    "ZayaForCausalLM": ("zaya", "ZayaForCausalLM"),',
            ),
        ],
    )
)

# 2. cca_state_shape in MambaStateShapeCalculator
mamba_path = os.path.join(VLLM_PATH, "v1", "worker", "gpu_model_runner.py")
# Might be in different location in 0.20.2


# Read the registry file
with open(registry_path) as f:
    content = f.read()

if '"ZayaForCausalLM"' not in content:
    # Find the right insertion point
    old = "_TEXT_GENERATION_MODELS = {"
    new = '_TEXT_GENERATION_MODELS = {\n    "ZayaForCausalLM": ("zaya", "ZayaForCausalLM"),'
    content = content.replace(old, new)
    with open(registry_path, "w") as f:
        f.write(content)
    print("PATCH 1: ZayaForCausalLM registered in ModelRegistry")
else:
    print("PATCH 1: Already registered")

# Patch 2: cca_state_shape
# Search for the file containing MambaStateShapeCalculator
for root, dirs, files in os.walk(VLLM_PATH):
    for f in files:
        if f.endswith(".py"):
            fpath = os.path.join(root, f)
            try:
                with open(fpath) as fh:
                    c = fh.read()
            except Exception:
                continue

            if "class MambaStateShapeCalculator" in c and "cca_state_shape" not in c:
                # Add cca_state_shape method
                new_method = '''
    def cca_state_shape(self, model_config, num_seqs, max_model_len):
        """CCA state shape for Zaya models."""
        num_layers = model_config.get_num_layers(model_config.get_hf_config())
        return (num_layers, num_seqs, max_model_len, model_config.get_hidden_size())
'''
                # Insert before the last method or at end of class
                old_str = "class MambaStateShapeCalculator"
                idx = c.find(old_str)
                if idx >= 0:
                    # Find the class closing (rough)
                    class_end = c.rfind("\n    def ", idx)
                    if class_end < 0:
                        class_end = len(c)
                    c = c[:class_end] + new_method + c[class_end:]
                    with open(fpath, "w") as fh:
                        fh.write(c)
                    print(f"PATCH 2: cca_state_shape added to {fpath}")
                break
            elif "class MambaStateShapeCalculator" in c and "cca_state_shape" in c:
                print("PATCH 2: Already patched")
                break

# Patch 3: cca_state_dtype
for root, dirs, files in os.walk(VLLM_PATH):
    for f in files:
        if f.endswith(".py"):
            fpath = os.path.join(root, f)
            try:
                with open(fpath) as fh:
                    c = fh.read()
            except Exception:
                continue

            if "class MambaStateDtypeCalculator" in c and "cca_state_dtype" not in c:
                new_method = '''
    def cca_state_dtype(self, model_config):
        """CCA state dtype for Zaya models."""
        return model_config.dtype
'''
                idx = c.find("class MambaStateDtypeCalculator")
                if idx >= 0:
                    class_end = c.rfind("\n    def ", idx)
                    if class_end < 0:
                        class_end = len(c)
                    c = c[:class_end] + new_method + c[class_end:]
                    with open(fpath, "w") as fh:
                        fh.write(c)
                    print(f"PATCH 3: cca_state_dtype added to {fpath}")
                break
            elif "class MambaStateDtypeCalculator" in c and "cca_state_dtype" in c:
                print("PATCH 3: Already patched")
                break

print("Done.")
