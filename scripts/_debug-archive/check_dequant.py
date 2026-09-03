"""Verify NVFP4 dequant produces correct weights for group_size=64."""
from __future__ import annotations

import safetensors.torch as st
import torch
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
from compressed_tensors.quantization.lifecycle.forward import dequantize

CT_PATH = r"C:\Users\ttimm\Documents\Project Portfolio\zaya1-godspeed\zaya1-8b-nvfp4-ct"

# Load CT model weights
ct_state = st.load_file(f"{CT_PATH}/model.safetensors", device="cpu")

# Load original model lm_head
from transformers import AutoModelForCausalLM

print("Loading original BF16 model lm_head...")
orig = AutoModelForCausalLM.from_pretrained(
    "Zyphra/ZAYA1-8B", torch_dtype=torch.bfloat16, device_map="cpu",
    trust_remote_code=True, low_cpu_mem_usage=True,
)

# Check o_proj from layer 0
layer_name = "model.layers.0.self_attn.o_proj"
print(f"\nChecking {layer_name}...")
wq = ct_state[f"{layer_name}.weight_packed"]
ws = ct_state[f"{layer_name}.weight_scale"]
wgs = ct_state.get(f"{layer_name}.weight_global_scale")

m, nh = wq.shape  # [out, in//2]
n = nh * 2         # unpacked input features

print(f"  packed shape: {list(wq.shape)} (uint8)")
print(f"  scale shape:  {list(ws.shape)} (FP8)")
print(f"  global_scale: {list(wgs.shape) if wgs is not None else 'None'}")
print(f"  group_size = {n // ws.shape[1]}")

# Dequantize
w = unpack_fp4_from_uint8(wq, m, n)
print(f"  unpacked shape: {list(w.shape)} (int)")

# Convert scale to float
ws_f = ws.float()
print(f"  scale range: [{ws_f.min():.6f}, {ws_f.max():.6f}]")

# Dequantize with global scale
w_deq = dequantize(
    x_q=w,
    scale=ws_f,
    global_scale=wgs.float() if wgs is not None else None,
    dtype=torch.float32,
)
print(f"  dequantized shape: {list(w_deq.shape)} (float32)")
print(f"  dequantized range: [{w_deq.min():.2f}, {w_deq.max():.2f}]")

# Compare against original
orig_w = orig.model.layers[0].self_attn.o_proj.weight.data
error = (w_deq.to(torch.float32) - orig_w.to(torch.float32)).abs().mean().item()
rel_err = error / orig_w.abs().mean().item() * 100
print(f"\n  vs original: mean_abs_error={error:.6f}, relative={rel_err:.2f}%")

# Check for NaN/Inf
print(f"  NaN: {torch.isnan(w_deq).any().item()}, Inf: {torch.isinf(w_deq).any().item()}")

# Also check a MoE layer
moe_name = "model.layers.1.zaya_block.experts.local_experts.0.linear_fc1"
print(f"\nChecking MoE {moe_name}...")
wq = ct_state[f"{moe_name}.weight_packed"]
ws = ct_state[f"{moe_name}.weight_scale"]
m, nh = wq.shape
n = nh * 2
w = unpack_fp4_from_uint8(wq, m, n)
w_deq = dequantize(x_q=w, scale=ws.float(), dtype=torch.float32)
print(f"  dequantized shape: {list(w_deq.shape)}")
print(f"  dequantized range: [{w_deq.min():.2f}, {w_deq.max():.2f}]")
print(f"  NaN: {torch.isnan(w_deq).any().item()}, Inf: {torch.isinf(w_deq).any().item()}")
