#!/usr/bin/env python3
"""Extract MOE_TARGET from session 2 fix script and write to vllm-src."""
import re
from pathlib import Path

fix_script = Path("/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/scripts/wsl_fix_nvfp4_text_gen.py")
target_file = Path("/home/ttimm/vllm-src/vllm/model_executor/layers/quantization/"
                    "compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py")

content = fix_script.read_text()
moe_start = content.find("MOE_TARGET = '''") + len("MOE_TARGET = '''")
moe_end = content.rfind("'''\n\n\ndef fix_moe")
moe_content = content[moe_start:moe_end]

print(f"Extracted {len(moe_content.split(chr(10)))} lines")
target_file.write_text(moe_content)
print(f"Written to {target_file}")
print("Done.")
