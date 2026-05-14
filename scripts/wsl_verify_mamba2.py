import sys, inspect
sys.path.insert(0, "/root/vllm-ct-env/lib/python3.12/site-packages")
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
print("File:", inspect.getfile(MambaStateShapeCalculator))
print("Has cca_state_shape:", hasattr(MambaStateShapeCalculator, "cca_state_shape"))
# Force reload
import importlib, vllm.model_executor.layers.mamba.mamba_utils as mu
importlib.reload(mu)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator as MSC2
print("After reload:", hasattr(MSC2, "cca_state_shape"))
