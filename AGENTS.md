# AGENTS.md — zaya1-godspeed

## Project Overview

Fine-tuning Zyphra's ZAYA1-8B (760M active / 8.4B total MoE) for agentic multi-turn
tool calling using teacher-distilled SFT+GRPO. Teacher: DeepSeek V4 Pro via NVIDIA NIM.
See `RESEARCH.md` for the full research document.

## Key Rules

### Tool-call format (CRITICAL)
ZAYA1-8B uses `<zyphra_tool_call>` / `<zyphra_tool_response>` tags (tokenizer IDs 101-104),
NOT `<tool_call>`. If you see `<tool_call>` anywhere, it's a bug. This was verified against
the official `Zyphra/ZAYA1-8B` tokenizer_config.json and `Zyphra/vllm` zaya_tool_parser.py.

### Zyphra fork transforms
This project depends on the Zyphra transformers fork (@ zaya1 branch) and vLLM fork
(@ zaya1-pr branch). The patches in `patches/` monkey-patch missing ecosystem features.
PR submitted: https://github.com/Zyphra/transformers/pull/2

### PEFT targets
Only target attention projections: `o_proj`, `linear_q`, `linear_k`, `val_proj1`, `val_proj2`.
Never target expert weights (SequentialMLP). This preserves ZAYA's MoE routing, EDA,
and MOD skip expert. Use `ensure_weight_tying=True` because `tie_word_embeddings=True`.

### Training constraints
- Epochs: 1-2 max. Hard stop if AIME degrades >5% from 89.1%.
- VRAM budget: ~9.5-11.5 GB (QLoRA on 16 GB). NF4 base model uses ~7.2 GB + double quantization saves ~0.36 GB.
- Chat template: `chat_template_kwargs={"enable_thinking": True}` required.
- GRPO uses `loss_type="dapo"`, `scale_rewards="batch"`, `num_generations=4`.
- QLoRA quant: NF4 with double quantization (`bnb_4bit_use_double_quant: true`). Bitsandbytes does not officially support 4-bit on MoE `nn.Parameter` tensors — LoRA on attention projections only (not experts) avoids this.

### Known limitations
- QLoRA + MoE: bitsandbytes 4-bit quantization is not validated for MoE architectures. Our LoRA targets attention projections only, not expert weights, so training is safe but monitor for quality issues.
- NF4 dequant on CCA attention: documented as broken in `COMPATIBILITY.md` — inference via NF4+transformers produces garbage output. Use vLLM for inference.
- Unsloth MoE Triton kernels: would provide 2-12x faster training but ZAYA's `SequentialMLP` architecture isn't tested with Unsloth yet. Worth trying `FastLanguageModel.from_pretrained("Zyphra/ZAYA1-8B")` when GPU is available.
- Double quantization: added May 2026 — saves ~0.36 GB per QLoRA paper benchmarks (was the difference between OOM and success on 16GB GPUs).

### Upstream boundaries (DO NOT CHANGE)
These ZAYA innovations must not be touched:
- CCA attention (compressed convolutional attention)
- MOD skip expert (Mixture-of-Depths)
- EDA routing (depth-wise averaging)
- Fused bias+SwiGLU custom autograd
- FP32 residual accumulation
- Dual time-stream values (val_proj1 + val_proj2)

### Quality gates for training data
1. Mechanical verify hook (exit_code=0)
2. Jaccard tool selection ≥0.7
3. Zero dangerous command flags
4. Zero schema validation errors
5. Token budget ≤4096

### File responsibilities
| File | Role | Change policy |
|------|------|---------------|
| `scripts/remap_to_zaya.py` | Godspeed JSONL → ZAYA ChatML | Must preserve `<zyphra_tool_call>` format |
| `scripts/train.py` | QLoRA SFT via TRL SFTTrainer | Uses SFTConfig, assistant_only_loss=True |
| `scripts/train_grpo.py` | GRPO policy improvement | Uses GRPOConfig, dapo loss, vLLM colocate |
| `scripts/mutate_tasks.py` | 200+ variant tasks | 6 mutation types, 30% OOD minimum |
| `data/generate.py` | Godspeed → ChatML extraction | Legacy — prefer remap_to_zaya.py |
| `configs/lora_tool_call.yaml` | Single source of truth for hyperparams | rsLoRA, chunked_nll, Liger Kernel, double quant |
| `patches/` | Runtime monkey-patches + upstream PR docs | `apply_zaya_patches.py` auto-runs in train.py/train_grpo.py |
| `tests/` | 100 unit tests | Run before any commit |

## Development workflow
1. `uv sync --dev` to install deps
2. `uv run ruff check .` for linting (0 errors required)
3. `uv run pytest tests/` for tests (100 required, 0 failures)
