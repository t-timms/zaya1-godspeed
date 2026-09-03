"""Quick check: original vs dequantized weight magnitude."""
from __future__ import annotations

import safetensors.torch as st
import torch
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
from compressed_tensors.quantization.lifecycle.forward import dequantize

CT_PATH = r"C:\Users\ttimm\Documents\Project Portfolio\zaya1-godspeed\zaya1-8b-nvfp4-ct"
ct_state = st.load_file(f"{CT_PATH}/model.safetensors", device="cpu")
from transformers import AutoModelForCausalLM

orig = AutoModelForCausalLM.from_pretrained(
    "Zyphra/ZAYA1-8B", dtype=torch.bfloat16, device_map="cpu",
    trust_remote_code=True, low_cpu_mem_usage=True,
)

# Check a few layers
for layer_name in [
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.self_attn.qkv.linear_q",
    "lm_head",
]:
    wq = ct_state.get(f"{layer_name}.weight_packed")
    if wq is None:
        print(f"{layer_name}: SKIP (not quantized)")
        continue
    ws = ct_state[f"{layer_name}.weight_scale"]
    wgs = ct_state.get(f"{layer_name}.weight_global_scale")
    m, nh = wq.shape
    n = nh * 2
    w = unpack_fp4_from_uint8(wq, m, n)
    w_deq = dequantize(x_q=w, scale=ws.float(), dtype=torch.float32)

    if layer_name == "lm_head":
        orig_w = orig.lm_head.weight.data
    elif "linear_q" in layer_name:
        orig_w = orig.model.layers[0].self_attn.qkv.linear_q.weight.data
    else:
        orig_w = orig.model.layers[0].self_attn.o_proj.weight.data

    err = (w_deq.to(torch.float32) - orig_w.to(torch.float32)).abs().mean().item()
    rel = err / orig_w.abs().mean().item() * 100
    print(f"{layer_name}:")
    print(f"  deq range=[{w_deq.min():.4f}, {w_deq.max():.4f}]")
    print(f"  orig range=[{orig_w.min():.4f}, {orig_w.max():.4f}]")
    print(f"  mean_abs_err={err:.6f}, relative={rel:.2f}%")
    print(f"  gs={'present' if wgs is not None else 'MISSING'}")
    print()

# Also check the attention weights specifically - do they have global_scale?
print("Checking for weight_global_scale in checkpoint...")
gs_keys = [k for k in ct_state if "weight_global_scale" in k]
print(f"  Found {len(gs_keys)} weight_global_scale keys")
if gs_keys:
    for k in gs_keys[:3]:
        print(f"  {k}: shape={list(ct_state[k].shape)}")
