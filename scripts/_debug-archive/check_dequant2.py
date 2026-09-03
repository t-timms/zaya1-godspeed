"""Compare dequantize vs NVFP4PackedCompressor.decompress for group_size=64."""
from __future__ import annotations

import safetensors.torch as st
import torch
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy, QuantizationType
from compressed_tensors.quantization.lifecycle.forward import dequantize

CT_PATH = r"C:\Users\ttimm\Documents\Project Portfolio\zaya1-godspeed\zaya1-8b-nvfp4-ct"
ct_state = st.load_file(f"{CT_PATH}/model.safetensors", device="cpu")

layer_name = "model.layers.0.self_attn.o_proj"
wq = ct_state[f"{layer_name}.weight_packed"]
ws = ct_state[f"{layer_name}.weight_scale"]
wgs = ct_state.get(f"{layer_name}.weight_global_scale")
print(f"weight_packed: {list(wq.shape)}, weight_scale: {list(ws.shape)}")
print(f"weight_global_scale: {list(wgs.shape) if wgs is not None else 'MISSING'}")

m, nh = wq.shape
n = nh * 2

# Method 1: Basic unpack + dequantize (what the fallback uses)
w1 = unpack_fp4_from_uint8(wq, m, n)
w1_deq = dequantize(x_q=w1, scale=ws.float(), dtype=torch.float32)
print(f"\nMethod 1 (dequantize): range=[{w1_deq.min():.4f}, {w1_deq.max():.4f}]")

# Method 2: NVFP4PackedCompressor.decompress (proper CT API)
comp_state = {
    "weight_packed": wq,
    "weight_scale": ws,
}
if wgs is not None:
    comp_state["weight_global_scale"] = wgs

# Create scheme matching the config (group_size=64)
scheme = {
    "weights": QuantizationArgs(
        num_bits=4,
        type=QuantizationType.FLOAT,
        strategy=QuantizationStrategy.GROUP,
        group_size=64,
        symmetric=True,
        dynamic=False,
    )
}
decomp = NVFP4PackedCompressor.decompress(comp_state, scheme)
w2_deq = decomp["weight"]
print(f"Method 2 (NVFP4PackedCompressor): range=[{w2_deq.min():.4f}, {w2_deq.max():.4f}]")

# Method 3: dequantize with global_scale=1.0 (since we don't have wgs)
w3 = unpack_fp4_from_uint8(wq, m, n)
w3_deq = dequantize(x_q=w3, scale=ws.float(), global_scale=torch.ones(m, dtype=torch.float32), dtype=torch.float32)
print(f"Method 3 (dequantize + ones gs): range=[{w3_deq.min():.4f}, {w3_deq.max():.4f}]")

# Compare against original
from transformers import AutoModelForCausalLM

orig = AutoModelForCausalLM.from_pretrained(
    "Zyphra/ZAYA1-8B", torch_dtype=torch.bfloat16, device_map="cpu",
    trust_remote_code=True, low_cpu_mem_usage=True,
)
orig_w = orig.model.layers[0].self_attn.o_proj.weight.data

for label, w in [("Method 1", w1_deq), ("Method 2", w2_deq), ("Method 3", w3_deq)]:
    err = (w.to(torch.float32) - orig_w.to(torch.float32)).abs().mean().item()
    rel = err / orig_w.abs().mean().item() * 100
    print(f"  {label}: mean_abs_err={err:.6f}, relative={rel:.2f}%")

# Also check the scale as-is
print(f"\nScale stats: min={ws.float().min():.6f}, max={ws.float().max():.6f}")
# The scales seem very small. Are these the actual per-group scales?
# For NVFP4, the scale should be around 0.01-0.1 typically.

# Check: what does ws look like for the first output channel?
print(f"First output channel scales: {ws[0].float()}")
