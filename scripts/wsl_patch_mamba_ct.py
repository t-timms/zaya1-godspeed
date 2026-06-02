p = "/root/vllm-ct-env/lib/python3.12/site-packages/vllm"

import os  # noqa: E402

# Patch 1: cca_state_shape
for root, dirs, files in os.walk(p):
    for f in files:
        if f.endswith(".py"):
            fpath = os.path.join(root, f)
            try:
                with open(fpath) as fh:
                    c = fh.read()
            except Exception:
                continue

            if "class MambaStateShapeCalculator" in c and "cca_state_shape" not in c:
                new_method = """
    @classmethod
    def cca_state_shape(cls, model_config, num_seqs, max_model_len):
        return (model_config.get_num_layers(model_config.get_hf_config()),
                num_seqs, max_model_len, model_config.get_hidden_size())
"""
                idx = c.find("class MambaStateShapeCalculator")
                end = c.rfind("\n    @", idx)
                if end < 0:
                    end = c.rfind("\n    def ", idx)
                if end < 0:
                    end = len(c)
                c = c[:end] + new_method + c[end:]
                with open(fpath, "w") as fh:
                    fh.write(c)
                print("PATCH 2: cca_state_shape added to", fpath)
                break
            elif "class MambaStateShapeCalculator" in c and "cca_state_shape" in c:
                print("PATCH 2: Already patched")
                break

# Patch 2: cca_state_dtype
for root, dirs, files in os.walk(p):
    for f in files:
        if f.endswith(".py"):
            fpath = os.path.join(root, f)
            try:
                with open(fpath) as fh:
                    c = fh.read()
            except Exception:
                continue

            if "class MambaStateDtypeCalculator" in c and "cca_state_dtype" not in c:
                new_method = """
    @classmethod
    def cca_state_dtype(cls, model_config):
        return model_config.dtype
"""
                idx = c.find("class MambaStateDtypeCalculator")
                end = c.rfind("\n    @", idx)
                if end < 0:
                    end = c.rfind("\n    def ", idx)
                if end < 0:
                    end = len(c)
                c = c[:end] + new_method + c[end:]
                with open(fpath, "w") as fh:
                    fh.write(c)
                print("PATCH 3: cca_state_dtype added to", fpath)
                break
            elif "class MambaStateDtypeCalculator" in c and "cca_state_dtype" in c:
                print("PATCH 3: Already patched")
                break
