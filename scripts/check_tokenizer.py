"""Check if Zyphra transformers fork includes Zaya tokenizer."""
import sys

sys.path.insert(0, "/home/ttimm/vllm-env/lib/python3.12/site-packages")

from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING

zaya_keys = [k for k in TOKENIZER_MAPPING.keys() if "zaya" in str(k).lower()]
print(f"Zaya tokenizer keys: {zaya_keys}")

# Also check CONFIG_MAPPING
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

zaya_configs = [k for k in CONFIG_MAPPING.keys() if "zaya" in str(k).lower()]
print(f"Zaya config keys: {zaya_configs}")

# Check MODEL_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

zaya_models = [k for k in MODEL_FOR_CAUSAL_LM_MAPPING.keys() if "zaya" in str(k).lower()]
print(f"Zaya model keys: {zaya_models}")

# Check if ZayaTokenizerFast exists
try:
    from transformers.models.zaya.tokenization_zaya import ZayaTokenizerFast
    print(f"ZayaTokenizerFast: {ZayaTokenizerFast}")
except ImportError as e:
    print(f"ZayaTokenizerFast not found: {e}")

# Also try to load config
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct-gs16", trust_remote_code=True)
    print(f"Config type: {type(config).__name__}")
except Exception as e:
    print(f"Config load failed: {e}")
