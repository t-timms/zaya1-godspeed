# ZAYA1-8B Compatibility Analysis

## Architecture

| Property | Value |
|----------|-------|
| Architecture | `ZayaForCausalLM` (custom) |
| Total params | 8.4B |
| Active params | 760M (MoE, 16 experts, top-1 routing) |
| Hidden size | 2048 |
| Num layers | 80 (40 attention + 40 MoE, interleaved) |
| Attention heads | 16 query, 2 KV (CCA — Compressed Convolutional Attention) |
| Context window | 131,072 |
| Special tokens | Qwen-style `<|im_start|>` / `<|im_end|>` |
| Thinking mode | `<think>` blocks, always-on in chat template |
| License | Apache 2.0 |

## PEFT Compatibility Gate ✓ PASSED

```
Model loaded: 7.24 GB VRAM (NF4 quantized)
LoRA attached: 8,192,000 trainable params (0.0926%)
VRAM with LoRA: 7.24 GB (+0.00 GB)
Gradient flow: OK (400 params with gradients)
```

**Target modules**: `o_proj`, `linear_q`, `linear_k`, `val_proj1`, `val_proj2`

These are the dense attention projections on the 40 attention layers
(layers 0, 2, 4, ..., 78). The 40 MoE layers (1, 3, 5, ..., 79) are interleaved.

LoRA targets the shared attention backbone rather than the expert weights,
per best practice for MoE fine-tuning where expert weights specialize in
domain knowledge and attention projections handle formatting/instruction.

**VRAM budget for QLoRA training on 16 GB GPU:**

| Method | VRAM |
|--------|------|
| Full fine-tune | ~40 GB+ |
| LoRA (bf16) | ~20-24 GB |
| QLoRA (4-bit + LoRA) | ~12-16 GB |
| QLoRA + gradient checkpointing | ~10-13 GB |

## ZAYA XML Tool-Call Format

ZAYA1-8B uses a JSON-inside-XML format via vLLM's `--tool-call-parser zaya_xml`.
Uses special tokens from the tokenizer (IDs 101-104):

```xml
<zyphra_tool_call>{"name": "tool_name", "arguments": {...}}</zyphra_tool_call>
<zyphra_tool_response>result text here</zyphra_tool_response>
```

**Critical distinction**: This differs from Godspeed's native Qwen3-Coder XML format
(`<function=name>`) and from OpenAI-standard `tool_calls` JSON. Training data MUST
use the ZAYA XML format with the correct `<zyphra_tool_call>` / `<zyphra_tool_response>`
tags. A remapper (`scripts/remap_to_zaya.py`) converts Godspeed
conversation JSONL to this format with quality gates applied.

The ZAYA1-8B chat template uses Qwen-style tokens:
```
<|im_start|>system
You are Godspeed...
<|im_end|>
<|im_start|>user
Fix the bug in this codebase...
<|im_end|>
<|im_start|>assistant
<think>
Let me explore the codebase first.
</think>

<zyphra_tool_call>{"name": "glob_search", "arguments": {"pattern": "**/*.py"}}</zyphra_tool_call>
<|im_end|>
<|im_start|>tool
<zyphra_tool_response>src/main.py
src/config.py
src/models.py</zyphra_tool_response>
<|im_end|>
```

**No external `quantization_config` needed**: When loading a pre-quantized
model, use the model's stored config:

```python
model = AutoModelForCausalLM.from_pretrained(
    "Zyphra/ZAYA1-8B",
    device_map="auto",
    trust_remote_code=True,
)
```

## Agentic Gap (Per Technical Report)

ZAYA1-8B's technical report confirms the agentic gap is intentional:

> "ZAYA1-8B does not include a dedicated multi-turn agentic RL stage in this
> release. We include some supervised agent, tool, and SWE traces during SFT,
> but the RL cascade is primarily optimized for verifiable reasoning, math,
> code, and instruction-following behavior."

This means ZAYA1-8B already has some tool-use exposure from its supervised
fine-tuning stage. The fine-tuning task is distribution extension, not cold start.

## Inference Paths

| Path | Status | Speed | Notes |
|------|--------|-------|-------|
| **vLLM (Zyphra fork), FP8** | 🟡 Experimental | ~4 tok/s (1st req, JIT) | 8.76 GB model. Tested May 11: loads and serves, output quality **unverified**. Needs `--reasoning-parser qwen3`. First request slow (Triton JIT), subsequent should be faster. |
| **vLLM (Zyphra fork), bf16** | ❌ OOM | N/A | 16.48 GB model exceeds 15.92 GB GPU. No room for KV cache. |
| **NF4 + transformers** | ❌ Broken | 3 tok/s | Garbage output. Bitsandbytes NF4 dequant incompatible with Zaya CCA attention. |
| **GGUF / llama.cpp** | ❌ Blocked | N/A | Zaya architecture not supported ([llama.cpp#22776](https://github.com/ggml-org/llama.cpp/issues/22776)). |
| **NVFP4 (llama.cpp quantizer)** | ⬜ Planned | TBD | Our own 4 GB quantized model on Blackwell sm_120. See `ROADMAP.md`. |
| **Zyphra Cloud** | Available | API latency | Serverless endpoint at cloud.zyphra.com. Needs API key. |

### vLLM deployment

```bash
# Full context (24 GB+ cards)
vllm serve Zyphra/ZAYA1-8B --port 8010 \
    --mamba-cache-dtype float32 --dtype bfloat16 \
    --reasoning-parser qwen3 --enable-auto-tool-choice \
    --tool-call-parser zaya_xml \
    --max-num-seqs 2 --max-model-len 48000

# 16 GB cards — FP8 quantization (8.76 GB model, 5.37 GB KV cache)
vllm serve Zyphra/ZAYA1-8B --port 8010 \
    --quantization fp8 --dtype bfloat16 \
    --reasoning-parser qwen3 --enable-auto-tool-choice \
    --tool-call-parser zaya_xml \
    --max-num-seqs 1 --max-model-len 4096 \
    --trust-remote-code --enforce-eager
```

Requires the Zyphra vLLM fork built in WSL (see `scripts/build_vllm_detached.sh`).

## DeepSeek V4 Pro Teacher (NVIDIA NIM)

Trajectory generation uses DeepSeek V4 Pro via NVIDIA NIM as the teacher model:

- **Model ID**: `nvidia_nim/deepseek-ai/deepseek-v4-pro`
- **SWE-bench Verified**: 80.6% | **LiveCodeBench**: 93.5% | **Codeforces**: 3,206
- **Architecture**: MoE, 1.6T total / 49B active per token, 1M context
- **Availability**: NVIDIA NIM free tier (R&D), 4 API keys for rate limit rotation
- **Rate limit**: ~30 RPM per key → ~120 RPM effective with rotation

Godspeed routes to NIM via its existing `nvidia_nim/` provider prefix
(confirmed working in benchmark shootout, April 2026).

## Training Strategy

### Objective

Fine-tune ZAYA1-8B for structured multi-step tool calling with Godspeed's
tool schema. Target the agentic failure mode: the model has strong reasoning
but was not trained for iterative tool use.

### Pipeline

1. **Teach** — Run Godspeed headless with DeepSeek V4 Pro against 200+ mutated tasks
2. **Filter** — `remap_to_zaya.py` applies 5 quality gates (exit code, Jaccard, dangerous commands, schema errors, token budget)
3. **SFT** — QLoRA training on 300–500 verified trajectories (1–2 epochs)
4. **GRPO** — Policy improvement with verifiable rewards (mechanical verify as primary signal)

### LoRA Targeting

Target attention projections ONLY (`o_proj`, `linear_q`, `linear_k`, `val_proj1`,
`val_proj2`). These dense shared layers control formatting and instruction
following, not domain-specific reasoning (which lives in the 1280 expert weights).

### Training Config

See `configs/lora_tool_call.yaml`.

## Upstream Code Audit (Zyphra/transformers @ zaya1, Zyphra/vllm @ zaya1-pr)

Audit performed May 10, 2026 against the modular source (`modular_zaya.py`) and
the vLLM tool parser (`zaya_tool_parser.py`).

### Already SOTA (no action needed)

| Feature | Implementation |
|---------|---------------|
| `torch.compile` fallback | Dynamically switches from `torch.jit.script` to `torch.compile` for torch ≥2.2 |
| FP8 activation storage | `fp8_input_store` option stores activations in float8 for backward pass |
| Fused bias+SwiGLU | `BiasSwiGLUFunction` custom autograd kernel avoids intermediate materialization |
| FP32 residual accumulation | `residual_in_fp32=True` for numerical stability in 80-layer model |
| Residual scaling | Per-layer learnable scale+bias on hidden states and residuals (DeepNet-style) |
| CCA attention | Depthwise+grouped conv1d QK mixing, L2-normalized QK, per-head temperatures |
| Dual time-stream values | `val_proj1` (current) + `val_proj2` (shifted) for richer value representation |
| EDA routing | Depth-wise averaging of router hidden states across MoE layers |
| MOD skip expert | Mixture-of-Depths: tokens can bypass MoE computation entirely |
| Three attention backends | `eager`, `sdpa`, `flash_attention_2` dispatch via config |
| PEFT-aware casting | FlashAttention detects PEFT fp32 layer norms and casts back to target dtype |

### Findings (potential improvements in upstream)

Compared against DeepSeek-V4 Pro (May 2026, 1.6T/49B MoE, 1M ctx), DeepSeek-V3.1
(`modeling_deepseek_v3.py`), and Qwen3-MoE (`modular_qwen3_moe.py`).

#### DeepSeek-V4 Pro SOTA Reference (Teacher Model)

| Property | DeepSeek V4 Pro | ZAYA1-8B |
|----------|----------------|----------|
| Total params | 1.6T | 8.4B |
| Active params | 49B | 760M |
| Context window | 1,000,000 | 131,072 |
| Precision | FP4+FP8 mixed | BF16 |
| Attention | CSA + HCA hybrid | CCA |
| Connections | mHC (manifold-constrained) | Residual scaling |
| Optimizer | Muon | Not specified |
| Post-training | On-policy distillation | 4-stage RL cascade |
| SWE Verified | 80.6% | N/A (BFCL-v4: 39.22) |
| Reasoning modes | Non-think / High / Max | Always-on thinking |

| # | Finding | Severity | Present in DeepSeek-V3 | Present in Qwen3-MoE |
|---|---------|----------|----------------------|---------------------|
| 1 | **No `output_router_logits` / aux loss** | **High** | Yes | Yes |
| 2 | **`SequentialMLP` not fused MoE** | **High** | Uses 3D `nn.Parameter` | Uses 3D `nn.Parameter` |
| 3 | **No `GradientCheckpointingLayer`** | Medium | Yes | Yes |
| 4 | **No `_can_compile_fullgraph = True`** | Medium | Yes | Yes |
| 5 | **No `_can_record_outputs`** | Medium | Yes | Yes |
| 6 | **No `logits_to_keep` support** | Medium | Yes | Yes |
| 7 | **No `@use_kernel_func_from_hub` for RoPE** | Low | Yes | Yes |
| 8 | **No `@use_experts_implementation`** | Medium | Yes | Yes |
| 9 | **No `_tied_weights_keys` class attribute** | Low | Yes | Yes |
| 10 | **No `_tp_plan` / `_pp_plan`** | Low | Yes | Yes |
| 11 | **No `_init_weights` override for MoE** | Medium | Yes | Yes |
| 12 | **No `_supports_flex_attn = True`** | Low | Yes | Yes |
| 13 | **Custom RMSNorm not hub-loaded** | Low | Yes | Yes |
| 14 | **`ZayaDecoderATTLayer` extends `nn.Module` not `GradientCheckpointingLayer`** | Medium | Yes | Yes |

#### Detailed gap analysis

**#1: Missing MoE aux loss (HIGH impact)**

ZayaConfig lacks `output_router_logits` and `router_aux_loss_coef`. ZayaForCausalLM does not compute
`load_balancing_loss_func()`. Both DeepSeek-V3 and Qwen3-MoE compute auxiliary loss to prevent
expert collapse. Without this, training (especially GRPO) cannot monitor or penalize routing imbalance.
The ZayaRouter computes logits internally but never exposes them to the model output.

**#2: SequentialMLP vs fused 3D expert weights (HIGH impact)**

ZAYA uses `SequentialMLP(nn.ModuleList([MLP, MLP, ...]))` — each expert is a separate `nn.Module`
processed in a Python `for` loop. DeepSeek-V3 and Qwen3-MoE store expert weights as fused 3D
`nn.Parameter` tensors (`gate_up_proj: [num_experts, 2*intermediate, hidden]`). The 3D approach:
- Uses `F.linear()` instead of `nn.Linear()` per expert (avoids Module overhead)
- Is compatible with `@use_experts_implementation` which dispatches to fused MoE kernels
- Reduces memory fragmentation from 16 separate Linear modules
- Enables vLLM fused MoE inference

**#3-5: Training infrastructure gaps (MEDIUM impact)**

Missing `GradientCheckpointingLayer` (DeepSeek/Qwen3 use this), `_can_compile_fullgraph` (enables
`torch.compile` full model graph), and `_can_record_outputs` (enables TRL trainer to capture router
logits for aux loss). ZayaDecoderATTLayer extends `nn.Module` directly.

**#6: Missing `logits_to_keep` (MEDIUM impact)**

During training, computing logits for the full sequence is wasteful. DeepSeek-V3 and Qwen3-MoE
support `logits_to_keep` to compute only the last N token logits.

**#11: Missing MoE weight initialization (MEDIUM impact)**

ZAYA relies on default PyTorch `Linear` init for router and expert weights. DeepSeek-V3 explicitly
initializes `DeepseekV3TopkRouter.weight` with `normal_(std=initializer_range)` and
`e_score_correction_bias` to zeros. Qwen3-MoE does the same for `Qwen3MoeTopKRouter.weight`
and `Qwen3MoeExperts.gate_up_proj/down_proj`.

### Impact on this experiment

**Finding #1 (aux loss)** is the most impactful for training. During QLoRA SFT, the base model
(including expert weights) is frozen, so expert collapse is not a concern. However, for GRPO
Stage 2, if router-related weights become trainable, expert utilization should be
logged manually since the model cannot compute aux loss.

**Finding #2 (fused MoE)** affects training throughput but not correctness. The vLLM Zyphra
fork already includes fused MoE for inference. Training-time fused MoE is not available
in the transformers fork (same limitation as DeepSeek-V3 naive experts in HF transformers).

**DeepSeek V4 Pro as teacher**: The teacher model vastly exceeds ZAYA1-8B (49B vs 760M
active params, 1M vs 131K context, FP4+FP8 vs BF16). Key design patterns that ZAYA's
next iteration could adopt from DeepSeek V4 Pro:
- **Manifold-constrained hyper-connections (mHC)**: More stable signal propagation than
  ZAYA's residual scaling for very deep models
- **On-policy distillation**: Independent domain expert SFT → unified consolidation,
  potentially superior to ZAYA's 4-stage sequential RL cascade for multi-domain tasks
- **Muon optimizer**: Faster convergence and training stability vs standard AdamW
- **Hybrid CSA+HCA attention**: More efficient long-context attention than CCA alone

## References

- [ZAYA1-8B Technical Report (arXiv 2605.05365)](https://arxiv.org/abs/2605.05365)
- [Zyphra Blog Post](https://www.zyphra.com/post/zaya1-8b)
- [Zyphra vLLM Fork](https://github.com/Zyphra/vllm/tree/zaya1-pr)
- [Zyphra Transformers Fork](https://github.com/Zyphra/transformers/tree/zaya1)
- [NVIDIA NIM — DeepSeek V4 Pro](https://build.nvidia.com/deepseek-ai/deepseek-v4-pro)
- [Godspeed Coding Agent](https://github.com/omnipotence-eth/godspeed-coding-agent)
