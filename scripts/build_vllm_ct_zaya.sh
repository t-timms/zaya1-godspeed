#!/bin/bash
# Professional build: Zaya CT NVFP4 vLLM server
# Reproducible, version-controlled, single-command

set -e

VENV="$HOME/vllm-ct-env"
VLLM_VER="0.20.2"
TORCH_VER="2.11.0+cu130"
TORCH_URL="https://download.pytorch.org/whl/cu130"
ZAYA_FORK="/tmp/zaya-vllm"

echo "=== Zaya CT NVFP4 vLLM Build ==="

# 1. Fresh venv
if [ -d "$VENV" ]; then
    echo "Removing old env..."
    rm -rf "$VENV"
fi
python3 -m venv "$VENV"
source "$VENV/bin/activate"

# 2. Core deps
echo "Installing torch $TORCH_VER..."
pip install --quiet torch==2.11.0 torchvision==0.26.0 --index-url "$TORCH_URL"

echo "Installing vLLM $VLLM_VER..."
pip install --quiet "vllm==$VLLM_VER"

# 3. Zyphra transformers fork
echo "Installing Zyphra transformers fork..."
pip install --quiet "transformers @ git+https://github.com/Zyphra/transformers.git@zaya1"

# 4. Zyphra vLLM Python overlay
echo "Applying Zyphra Python overlay..."
if [ ! -d "$ZAYA_FORK" ]; then
    git clone --depth 1 --branch zaya1-pr https://github.com/Zyphra/vllm.git "$ZAYA_FORK"
fi

SITE="$VENV/lib/python3.12/site-packages/vllm"
SRC="$ZAYA_FORK/vllm"

for f in \
    "model_executor/layers/mamba/cca.py" \
    "v1/attention/backends/cca_attn.py" \
    "model_executor/models/zaya.py" \
    "tool_parsers/zaya_tool_parser.py" \
    "transformers_utils/configs/zaya.py"
do
    mkdir -p "$(dirname "$SITE/$f")"
    cp "$SRC/$f" "$SITE/$f"
    echo "  $f"
done

# 5. Register ZayaForCausalLM
python3 << 'PYEOF'
import os, sys
SITE = os.environ.get("SITE", "/root/vllm-ct-env/lib/python3.12/site-packages/vllm")
reg = os.path.join(SITE, "model_executor", "models", "registry.py")
with open(reg) as f:
    r = f.read()
if '"ZayaForCausalLM"' not in r:
    r = r.replace('_TEXT_GENERATION_MODELS = {', '_TEXT_GENERATION_MODELS = {\n    "ZayaForCausalLM": ("zaya", "ZayaForCausalLM"),')
    with open(reg, 'w') as f:
        f.write(r)
    print("  ModelRegistry: ZayaForCausalLM registered")
PYEOF

# 6. Patch zaya.py for compressed_tensors MoE naming
python3 << 'PYEOF'
import os
SITE = os.environ.get("SITE", "/root/vllm-ct-env/lib/python3.12/site-packages/vllm")
zaya = os.path.join(SITE, "model_executor", "models", "zaya.py")
with open(zaya) as f:
    z = f.read()

# Fix f-string format bug
z = z.replace(
    'WARNING: key {chkpt_weight_name} not in params!',
    'WARNING: key %s not in params!'
)

# Fix w13_weight for compressed_tensors
old = 'param_name = f"{fused_moe_prefix}.w13_weight"\n                    param = params_dict[param_name]'
new = 'param_name = f"{fused_moe_prefix}.w13_weight"\n                    if param_name not in params_dict:\n                        param_name_packed = f"{param_name}_packed"\n                        if param_name_packed in params_dict:\n                            param_name = param_name_packed\n                    param = params_dict[param_name]'
z = z.replace(old, new)

# Fix w2_weight for compressed_tensors
old2 = 'param_name = f"{fused_moe_prefix}.w2_weight"\n                    param = params_dict[param_name]'
new2 = 'param_name = f"{fused_moe_prefix}.w2_weight"\n                    if param_name not in params_dict:\n                        param_name_packed = f"{param_name}_packed"\n                        if param_name_packed in params_dict:\n                            param_name = param_name_packed\n                    param = params_dict[param_name]'
z = z.replace(old2, new2)

with open(zaya, 'w') as f:
    f.write(z)
print("  zaya.py: CT MoE naming patch applied")
PYEOF

# 7. Verify
echo ""
echo "=== Verification ==="
python3 -c "
import torch; print(f'torch {torch.__version__} CUDA={torch.cuda.is_available()}')
import vllm; print(f'vllm {vllm.__version__}')
from vllm.model_executor.models.zaya import ZayaForCausalLM; print('ZayaForCausalLM: OK')
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_nvfp4 import CompressedTensorsW4A16Fp4; print('CT W4A16Fp4: OK')
"

echo ""
echo "=== Build Complete ==="
echo "To serve:"
echo "  source $VENV/bin/activate"
echo "  vllm serve /mnt/c/Users/ttimm/Documents/Project\\\\ Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct --port 8020 --dtype float16 --max-model-len 2048 --trust-remote-code --enforce-eager --max-num-seqs 1 --tokenizer Zyphra/ZAYA1-8B"
