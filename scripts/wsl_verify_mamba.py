import sys
sys.path.insert(0, "/root/vllm-ct-env/lib/python3.12/site-packages")
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
print("Has cca_state_shape:", hasattr(MambaStateShapeCalculator, "cca_state_shape"))
methods = [m for m in dir(MambaStateShapeCalculator) if "state" in m.lower() or "cca" in m.lower()]
print("Methods:", methods)
