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

## Inference Paths

| Path | Status | Speed | Notes |
|------|--------|-------|-------|
| **vLLM (Zyphra fork)** | Required | 50+ tok/s | Recommended by Zyphra. Supports `--tool-call-parser zaya_xml`. |
| **NF4 + transformers** | ❌ Broken | 3 tok/s | Garbage output. Bitsandbytes NF4 dequant incompatible with Zaya CCA attention. |
| **GGUF / llama.cpp** | ❌ Blocked | N/A | Zaya architecture not supported ([llama.cpp#22776](https://github.com/ggml-org/llama.cpp/issues/22776)). |
| **Zyphra Cloud** | Available | API latency | Serverless endpoint at cloud.zyphra.com. Needs API key. |

### Model loading with the Zyphra fork

The model requires the Zyphra transformers fork:

```bash
pip install "transformers @ git+https://github.com/Zyphra/transformers.git@zaya1"
```

**Critical**: Do NOT pass `quantization_config` when loading a pre-quantized
model. Use the model's stored config:

```python
# CORRECT — no external quantization_config
model = AutoModelForCausalLM.from_pretrained(
    "Zyphra/ZAYA1-8B",
    device_map="auto",
    trust_remote_code=True,
)
```

### vLLM deployment

```bash
vllm serve Zyphra/ZAYA1-8B --port 8010 \
    --mamba-cache-dtype float32 --dtype bfloat16 \
    --reasoning-parser qwen3 --enable-auto-tool-choice \
    --tool-call-parser zaya_xml \
    --max-num-seqs 2 --max-model-len 48000
```

For 16 GB cards, reduce `--max-model-len` to 24000–32000.

## Training Strategy

### Objective

Fine-tune ZAYA1-8B for structured multi-step tool calling with Godspeed's
tool schema. Target the agentic failure mode: the model has strong reasoning
but was not trained for iterative tool use.

### Data Pipeline

1. Run Godspeed with a strong API model (Claude/GPT) on the benchmark suite
2. Log all conversations + per-step reward annotations
3. Export successful tool-call trajectories
4. Format as instruction tuning data (system_prompt → plan → tool_call → result)

### LoRA Targeting

Target attention projections ONLY (`o_proj`, `linear_q`, `linear_k`, `val_proj1`,
`val_proj2`). These dense shared layers control formatting and instruction
following, not domain-specific reasoning (which lives in the 1280 expert weights).

### Training Config

See `configs/lora_tool_call.yaml`.

## References

- [ZAYA1-8B Technical Report](https://arxiv.org/abs/2605.05365)
- [Zyphra Blog Post](https://www.zyphra.com/post/zaya1-8b)
- [Zyphra vLLM Fork](https://github.com/Zyphra/vllm/tree/zaya1-pr)
- [Zyphra Transformers Fork](https://github.com/Zyphra/transformers/tree/zaya1)
- [Godspeed Coding Agent](https://github.com/omnipotence-eth/godspeed-coding-agent)
