# Research Document: Completing ZAYA1-8B with Agentic Multi-Turn Tool Calling

**Project**: zaya1-godspeed  
**Experiment Lead**: Tremayne Timms  
**Date**: May 10, 2026  
**Status**: Pre-training (pipeline validated, awaiting GPU availability)

---

## 1. Abstract

ZAYA1-8B (Zyphra, May 2026) achieves state-of-the-art reasoning benchmarks for its parameter class (760M active / 8.4B total, MoE). However, per its technical report, Zyphra deliberately skipped the multi-turn agentic RL stage. This project aims to close that agentic gap by fine-tuning ZAYA1-8B for structured multi-step tool calling using teacher-distilled SFT+GRPO with DeepSeek V4 Pro as the teacher model.

This document summarizes the compatibility analysis, upstream code audit, SOTA training configuration, and pipeline design as of May 2026.

---

## 2. Model Architecture Summary

| Property | Value |
|----------|-------|
| Architecture | `ZayaForCausalLM` (custom, Qwen-style chat template) |
| Total params | 8.4B |
| Active params | 760M (16 experts, top-1 routing, MoE) |
| Hidden size | 2048 |
| Layers | 80 (40 CCA attention + 40 MoE, interleaved) |
| Attention | Compressed Convolutional Attention (CCA): depthwise+grouped conv1d QK mixing, L2-normalized QK, per-head temperatures, dual time-stream values |
| Context window | 131,072 tokens (rope_theta=5,000,000) |
| Special tokens | `<|im_start|>`, `<|im_end|>` (Qwen-style), `<think>` blocks |
| Tool-call format | `<zyphra_tool_call>{"name":"...","arguments":{...}}</zyphra_tool_call>` (tokens 101-102) |
| Tool-response format | `<zyphra_tool_response>result</zyphra_tool_response>` (tokens 103-104) |
| Weight tying | `tie_word_embeddings=True` |
| License | Apache 2.0 |

---

## 3. Critical Discovery: Tool-Call Format Mismatch

### 3.1 The Finding

During the upstream code audit, we discovered that the project's training data format used incorrect tool-call boundary tags. The ZAYA1-8B tokenizer has dedicated single tokens for tool-call boundaries:

| Token ID | Content | Expected Behavior |
|----------|---------|-------------------|
| 101 | `<zyphra_tool_call>` | Single-token tool call start |
| 102 | `</zyphra_tool_call>` | Single-token tool call end |
| 103 | `<zyphra_tool_response>` | Single-token tool response start |
| 104 | `</zyphra_tool_response>` | Single-token tool response end |

The project was using `<tool_call>` / `</tool_call>` instead. These are not special tokens — the tokenizer would split them into 5-6 subword tokens each: `<`, `tool`, `_`, `call`, `>`. Training a model with the wrong boundary tokens would produce outputs that the vLLM `zaya_xml` parser cannot parse, and the model would not learn to use the dedicated single-token boundaries.

### 3.2 Resolution

All tool-call references across 8 files were updated to use `<zyphra_tool_call>` / `</zyphra_tool_call>`. Tool responses were updated to use `<zyphra_tool_response>` / `</zyphra_tool_response>` instead of the old `[tool_name result]\n` prefix. The remapper (`scripts/remap_to_zaya.py`) now produces training data that matches ZAYA1-8B's native output format.

### 3.3 Verification

The fix was verified by cross-referencing:
1. `Zyphra/ZAYA1-8B` tokenizer_config.json (tokens 101-104)
2. `Zyphra/vllm` zaya_tool_parser.py (line 59-60: `self.tool_call_start_token = "<zyphra_tool_call>"`)
3. 100 unit tests, all passing

---

## 4. Training Pipeline Design

### 4.1 Pipeline Architecture

```
200 mutated tasks → Godspeed headless (DeepSeek V4 Pro via NIM)
    → conversation JSONL → remap_to_zaya.py (5 quality gates)
    → train_zaya.jsonl → train.py (QLoRA SFT, SFTTrainer)
    → GRPO Stage 2 (train_grpo.py)
    → vLLM serve → Godspeed 20-task benchmark → BFCL-v4
```

### 4.2 Phase 1: Compatibility Gate ✓

- Verified Zyphra transformers fork loads ZAYA1-8B with PEFT LoRA
- Confirmed 8.2M trainable parameters (0.09% of total) targeting attention projections only
- Verified gradient flow through LoRA adapters
- Architecture documented in `COMPATIBILITY.md`

### 4.3 Phase 2: Inference Pipeline (Blocked)

- vLLM build and serve scripts ready
- Blocked by Windows desktop compositor consuming ~15.9 GB VRAM
- Zyphra Cloud API available as interim alternative

### 4.4 Phase 3-4: Teacher Trajectory Generation + Format Remapping

- 200+ mutated tasks generated via `scripts/mutate_tasks.py` (6 mutation types, 30% OOD)
- Quality gates applied during remapping:
  1. Mechanical verify hook (exit_code=0)
  2. Jaccard tool selection ≥0.7 vs expected tools
  3. Zero dangerous command flags
  4. Zero schema validation errors
  5. Maximum token budget (4096 estimated tokens)
- Format conversion: Godspeed JSONL → ZAYA ChatML with `<zyphra_tool_call>` tags
- Optional `--include-tools` flag adds TRL-compatible `tools` JSON schema column

### 4.5 Phase 5: Training (Ready)

#### Stage 1: QLoRA SFT (`scripts/train.py`)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Method | SFTTrainer (TRL) | Native conversational dataset, `assistant_only_loss` |
| Quantization | NF4 (bitsandbytes) | Fits on 16 GB GPU |
| LoRA rank | r=16 | Conservative for small dataset (300-500 examples) |
| LoRA targets | `o_proj`, `linear_q`, `linear_k`, `val_proj1`, `val_proj2` | Attention projections only, not expert weights |
| rsLoRA | Enabled | Rank-stabilized scaling: `alpha/√r` instead of `alpha/r` |
| Loss type | `chunked_nll` | 60% memory reduction via chunked cross-entropy |
| Liger Kernel | Enabled | 20% throughput increase, 60% memory reduction |
| assistant_only_loss | True | Only trains on assistant output tokens |
| Epochs | 2 | 1-2 epochs max per project constraints |
| Learning rate | 2e-4 | Higher LR for LoRA adapters |
| Gradient checkpointing | True | Reduces activation memory |
| VRAM budget | ~10-12 GB | Fits on 16 GB GPU |

#### Stage 2: GRPO (`scripts/train_grpo.py`)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Loss type | `dapo` | Token-level normalization, no response-length bias |
| Reward scaling | `batch` | Global std normalization, robust reward shaping |
| Num generations | 4 | 4 rollouts per prompt for advantage estimation |
| KL penalty | β=0.0 | KL penalty not needed per modern research |
| vLLM | Colocate mode | Fast generation on single GPU |
| Reward functions | Format (0.5), Schema (0.4), Length (0.1) | Primary: valid `<zyphra_tool_call>` XML + JSON |

### 4.6 Phase 6-7: Evaluation + Deployment (Not Started)

Target benchmarks:
- BFCL-v4: 39.22 → 50+ (SFT only), 55+ (SFT+GRPO)
- τ² (agentic): 43.12 → 65-75 (SFT+GRPO)
- Regression monitoring: AIME ≥84 (hard floor), LiveCodeBench ≥60

---

## 5. Upstream Code Audit (Zyphra)

### 5.1 Audited Repositories

- `Zyphra/transformers` @ zaya1 branch (`modular_zaya.py`, `configuration_zaya.py`)
- `Zyphra/vllm` @ zaya1-pr branch (`zaya_tool_parser.py`)
- `Zyphra/ZAYA1-8B` HuggingFace model card + `tokenizer_config.json`

### 5.2 SOTA Features Already Present

Zyphra's implementation includes several state-of-the-art optimizations:
- Dynamic `torch.compile`/`torch.jit.script` dispatch per PyTorch version
- FP8 activation storage for backward pass memory savings
- Fused bias+SwiGLU custom autograd kernels
- FP32 residual accumulation for numerical stability in 80-layer models
- Per-layer residual scaling (DeepNet-style)
- Dual time-stream value projections in CCA attention
- Depth-wise averaging (EDA) in MoE router
- Mixture-of-Depths (MOD) skip expert
- Three attention backends with PEFT-aware fp32 casting

### 5.3 Upstream Improvement Opportunities

| # | Finding | Impact |
|---|---------|--------|
| 1 | `debug_level` dead attributes (~30 per model) | Negligible |
| 2 | Unnecessary `.clone()` in router forward | Extra allocation per MoE layer |
| 3 | `SequentialMLP` no fused MoE kernels | 3-5x slower MoE training (vLLM has fused MoE for inference) |
| 4 | `o_proj` dimension hardcoded (`hidden_size // 2`) | Would break with non-default head configs |
| 5 | No auxiliary MoE load-balancing loss | Expert collapse risk if router weights are trained |
| 6 | `conv_states` pre-allocation for all 80 layers | ~400KB — negligible |

None of these are blocking for this experiment. Finding #5 (aux loss) is the most relevant — expert collapse is not a concern during QLoRA SFT since the base model (including expert weights) is frozen.

### 5.4 Chat Template Compatibility

ZAYA1-8B uses Qwen-style `<|im_start|>` / `<|im_end|>` tokens with `<think>` / `</think>` blocks. TRL auto-patches Qwen-family chat templates for `assistant_only_loss=True` and prefix-preservation for tool calls. Our configuration passes `chat_template_kwargs={"enable_thinking": True}` to match ZAYA1-8B's always-on thinking mode.

---

## 6. SOTA Training Configuration

### 6.1 Configuration Evolution

| Component | Original | Final (SOTA) | Source |
|-----------|----------|--------------|--------|
| Trainer | `Trainer` (vanilla) | `SFTTrainer` (TRL) | TRL docs |
| Loss masking | None | `assistant_only_loss=True` | TRL SFT docs |
| LoRA scaling | Standard (`α/r`) | rsLoRA (`α/√r`) | PEFT docs |
| Loss computation | Standard NLL | `chunked_nll` (60% mem reduction) | TRL SFT docs |
| Kernel library | None | Liger Kernel (20% throughput) | TRL Liger Kernel docs |
| Chat template | Default | `enable_thinking=True` | ZAYA1-8B docs |
| Weight tying | Not set | `ensure_weight_tying=True` | ZayaConfig |
| Epochs | 3 | 2 | Project docs constraint |
| GRPO loss | Not implemented | `dapo` (token-level norm) | DAPO paper |
| GRPO reward scaling | Not implemented | `batch` (global std) | TRL GRPO docs |
| Dataset format | Manual tokenization | Conversational (TRL native) | TRL Dataset Formats docs |

### 6.2 Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Catastrophic forgetting (AIME ↓) | QLoRA r=16, 2 epochs max. Hard stop if AIME <84 |
| Tool-call format corruption | All `<tool_call>` → `<zyphra_tool_call>` (critical fix applied) |
| Shopify OOD failure | 30% OOD tasks minimum in training data |
| Training data quality | 5 quality gates during remapping |
| Benchmark leakage | Tasks vs SWE-bench Verified IDs cross-check before eval |
| Expert collapse (GRPO) | Monitor expert utilization if router weights trainable |
| Format mismatch (teacher vs student) | Godspeed XML → ZAYA XML remapping with validation |

---

## 7. Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| `data/generate.py` | 23 | Parsing, filtering, truncation, session finding |
| `scripts/mutate_tasks.py` | 21 | All 6 mutation types, dedup, OOD ratio, seed |
| `scripts/remap_to_zaya.py` | 56 | Tool-call format, tool-response format, all 5 quality gates, Jaccard, token estimation, remap pipeline |
| **Total** | **100** | **All passing** |

---

## 8. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.5.0 | ML framework |
| transformers | Zyphra fork @ zaya1 | ZayaForCausalLM support |
| peft | ≥0.13.0 | LoRA adapters |
| bitsandbytes | ≥0.45.0 | NF4 quantization |
| trl | ≥0.24.0 | SFTTrainer, GRPOTrainer |
| datasets | ≥3.0.0 | HuggingFace datasets |
| pyyaml | ≥6.0 | Config parsing |
| vllm (optional) | Zyphra fork @ zaya1-pr | Inference server |
| unsloth (optional) | ≥2026.5.0 | Faster training |
| liger-kernel (optional) | ≥0.6.0 | Memory-efficient kernels |

---

## 9. References

1. ZAYA1-8B Technical Report. Washbourne et al. arXiv:2605.05365, May 2026.
2. TRL Documentation. HuggingFace. https://huggingface.co/docs/trl/
3. PEFT Documentation. HuggingFace. https://huggingface.co/docs/peft/
4. vLLM Documentation. https://docs.vllm.ai/
5. Unsloth Documentation. https://docs.unsloth.ai/
6. DeepSeekMath: Pushing the Limits of Mathematical Reasoning. Shao et al. arXiv:2402.03300, 2024.
7. DAPO: An Open-Source LLM RL System at Scale. Yu et al. arXiv:2503.14476, 2025.
8. Understanding R1-Zero-Like Training: A Critical Perspective. arXiv:2503.20783, 2025.
9. LoRA: Low-Rank Adaptation of Large Language Models. Hu et al. arXiv:2106.09685, 2021.
10. QLoRA: Efficient Finetuning of Quantized LLMs. Dettmers et al. arXiv:2305.14314, 2023.

---

## 10. Next Steps

1. Acquire GPU with ≥16 GB VRAM (RTX 5070 Ti or cloud equivalent)
2. Set up NVIDIA NIM credentials for DeepSeek V4 Pro teacher model
3. Run Phase 3: Generate teacher trajectories via Godspeed headless
4. Run Phase 4: Remap trajectories to ZAYA ChatML format
5. Run Phase 5 Stage 1: QLoRA SFT on verified trajectories
6. Evaluate baseline: Godspeed 20-task benchmark + AIME regression check
7. Run Phase 5 Stage 2: GRPO policy improvement
8. Run Phase 6: BFCL-v4, τ² evaluation
9. Run Phase 7: Deploy to Godspeed driver catalog
