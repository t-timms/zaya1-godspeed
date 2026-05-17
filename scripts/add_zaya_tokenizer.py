"""Add ZayaConfig -> Qwen2Tokenizer mapping to transformers."""
import sys

sys.path.insert(0, "/home/ttimm/vllm-env/lib")

from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.models.zaya.configuration_zaya import ZayaConfig

# Register the mapping
if ZayaConfig not in TOKENIZER_MAPPING:
    TOKENIZER_MAPPING[ZayaConfig] = Qwen2TokenizerFast
    print("Registered ZayaConfig -> Qwen2TokenizerFast")

    # Also register the slow tokenizer if available
    try:
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
        # Slow tokenizer is typically not mapped separately, just verifying
        print(f"Qwen2Tokenizer also available: {Qwen2Tokenizer}")
    except ImportError:
        pass
else:
    print("ZayaConfig already mapped:", TOKENIZER_MAPPING[ZayaConfig])
