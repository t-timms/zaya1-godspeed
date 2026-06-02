import inspect
import sys

sys.path.insert(0, "/root/vllm-ct-env/lib/python3.12/site-packages")
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

print("File:", inspect.getfile(MambaStateShapeCalculator))
print("Has cca_state_shape:", hasattr(MambaStateShapeCalculator, "cca_state_shape"))
# Force reload
import importlib  # noqa: E402

import vllm.model_executor.layers.mamba.mamba_utils as mu  # noqa: E402

importlib.reload(mu)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator as MSC2  # noqa: E402

print("After reload:", hasattr(MSC2, "cca_state_shape"))
