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
| **vLLM (Zyphra fork)** | Required | 50+ tok/s | Recommended by Zyphra. Supports `--tool-call-parser zaya_xml`. |
| **NF4 + transformers** | ❌ Broken | 3 tok/s | Garbage output. Bitsandbytes NF4 dequant incompatible with Zaya CCA attention. |
| **GGUF / llama.cpp** | ❌ Blocked | N/A | Zaya architecture not supported ([llama.cpp#22776](https://github.com/ggml-org/llama.cpp/issues/22776)). |
| **Zyphra Cloud** | Available | API latency | Serverless endpoint at cloud.zyphra.com. Needs API key. |

### vLLM deployment

```bash
# Full context (24 GB+ cards)
vllm serve Zyphra/ZAYA1-8B --port 8010 \
    --mamba-cache-dtype float32 --dtype bfloat16 \
    --reasoning-parser qwen3 --enable-auto-tool-choice \
    --tool-call-parser zaya_xml \
    --max-num-seqs 2 --max-model-len 48000

# 16 GB cards (use serve_zaya1.py)
python scripts/serve_zaya1.py --max-model-len 24000 --max-num-seqs 2
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

| # | Finding | Severity | Detail |
|---|---------|----------|--------|
| 1 | `debug_level` dead attributes | Low | Every submodule has `self.debug_level = N`. Never read in any conditional. ~30 unused attributes per model instance. |
| 2 | Unnecessary clone in router | Low | `router_hidden_states_next = hs[:, -S:].clone()` in `ZayaRouter.forward`. A view or `contiguous()` would avoid allocation. |
| 3 | SequentialMLP no fused MoE | Medium | 16 experts processed with Python `for` loop. Fused MoE kernels (vLLM `fused_moe`) would be 3-5x faster for training. Acceptable since vLLM fork already has fused MoE for inference. |
| 4 | `o_proj` dimension hardcoded | Low | `hidden_size // 2` assumed. Would break if `head_dim` or `num_attention_heads` changed from defaults (8 heads × 128 dim = 1024 = 2048/2). |
| 5 | No aux load-balancing loss | Medium | `output_router_logits` not wired in config. During training, MoE models typically need aux loss to prevent expert collapse. LoRA SFT freezes base model so experts don't collapse. For GRPO with trainable router weights, monitor expert utilization manually. |
| 6 | `conv_states` pre-allocation | Low | `torch.zeros(80, batch_size, 1280, 2)` for CCA cache. ~400KB — negligible. |

### Impact on this experiment

None of the upstream findings are blocking. The training pipeline targets only
attention projection layers (`o_proj`, `linear_q`, `linear_k`, `val_proj1`,
`val_proj2`) via QLoRA, leaving expert weights frozen. Expert collapse during
SFT is not a concern. For GRPO Stage 2, expert utilization should be logged.

Finding #5 (aux loss) is the most relevant — if future experiments train router
weights, add expert utilization logging to detect collapse.

## References

- [ZAYA1-8B Technical Report (arXiv 2605.05365)](https://arxiv.org/abs/2605.05365)
- [Zyphra Blog Post](https://www.zyphra.com/post/zaya1-8b)
- [Zyphra vLLM Fork](https://github.com/Zyphra/vllm/tree/zaya1-pr)
- [Zyphra Transformers Fork](https://github.com/Zyphra/transformers/tree/zaya1)
- [NVIDIA NIM — DeepSeek V4 Pro](https://build.nvidia.com/deepseek-ai/deepseek-v4-pro)
- [Godspeed Coding Agent](https://github.com/omnipotence-eth/godspeed-coding-agent)
