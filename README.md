# zaya1-godspeed

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![Unsloth](https://img.shields.io/badge/Unsloth-2026.5.2-6C4DFF?style=flat-square)](https://unsloth.ai)
[![TRL](https://img.shields.io/badge/TRL-v0.24.0-orange?style=flat-square)](https://github.com/huggingface/trl)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)](./LICENSE)

Fine-tuning Zyphra's ZAYA1-8B for structured multi-turn tool calling with the
[Godspeed coding agent](https://github.com/omnipotence-eth/godspeed-coding-agent).

ZAYA1-8B punches well above its weight on math and coding benchmarks
(760M active / 8.4B total params, MoE). Per its own [technical report](https://arxiv.org/abs/2605.05365)
(May 2026), Zyphra **deliberately skipped** the multi-turn agentic RL stage —
the agentic gap is a training omission, not a capability ceiling. This project
completes ZAYA1-8B by adding that missing stage, served from a 4-bit NVFP4
quantization on 16 GB consumer hardware.

## Status (May 2026)

- ✅ **NVFP4 Compressed-Tensors ZAYA1-8B** quantized at group_size=16 (5.04 GB)
- ✅ **First coherent text generation on Blackwell sm_120** via vLLM (May 14, session 2): "The capital of France is" → " Paris."; coherent BST explanation
- ✅ Serves at ~0.86 tok/s on RTX 5070 Ti (16 GB) using Path A on-the-fly Python dequant; bf16 inference dtype required
- ⬜ Stage 2 next: custom Blackwell NVFP4 Tensor Core CUDA kernel for >10× speedup
- ⬜ Phase 3+: Teacher trajectories, SFT, GRPO, BFCL-v4 / τ² evaluation

See [`RESEARCH.md`](./RESEARCH.md) §5.9–§5.10 for the five-bug debugging story and
[`ROADMAP.md`](./ROADMAP.md) for phase-by-phase status.

## Quick Start

```bash
uv sync
python scripts/test_peft.py   # verify PEFT compatibility
```

### Serve the NVFP4 model (WSL)

```bash
# In WSL: apply vLLM patches (idempotent), then run the smoke check.
source /home/ttimm/vllm-env/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
python3 scripts/wsl_fix_moe_scale_routing.py    # session 1
python3 scripts/wsl_fix_marlin_group_size.py    # session 1
python3 scripts/wsl_fix_nvfp4_text_gen.py       # session 2
bash    scripts/wsl_run_quick_check.sh          # dtype=bfloat16 required
```

## Project Structure

```
zaya1-godspeed/
├── scripts/
│   ├── train.py                    # QLoRA fine-tuning with TRL SFTTrainer
│   ├── serve_zaya1.py              # vLLM server launcher (n-gram spec, tool-call support)
│   ├── serve.py                    # vLLM server launcher for base model
│   ├── test_peft.py                # PEFT LoRA attach + gradient flow test
│   ├── remap_to_zaya.py            # Godspeed JSONL → ZAYA XML ChatML + quality gates
│   ├── mutate_tasks.py             # 20 benchmark tasks → 200+ variants with OOD coverage
│   └── build_vllm_detached.sh      # Reliable Zyphra vLLM fork build (WSL) (deprecated; use scripts/build_vllm_detached.sh)
├── configs/
│   └── lora_tool_call.yaml         # QLoRA config for tool-calling fine-tune
├── data/
│   └── generate.py                 # Convert Godspeed sessions to ChatML training data
├── COMPATIBILITY.md                # Architecture analysis, PEFT gate results, ZAYA XML format
├── ROADMAP.md                      # 7-phase project plan with status tracking
└── README.md
```

## Pipeline

```
200 mutated tasks → Godspeed headless (DeepSeek V4 Pro via NIM)
    → conversation JSONL → remap_to_zaya.py (quality gates)
    → train_zaya.jsonl → train.py (QLoRA SFT → GRPO)
    → vLLM serve → Godspeed 20-task benchmark → BFCL-v4
```

**Teacher model**: DeepSeek V4 Pro via NVIDIA NIM (`nvidia_nim/deepseek-ai/deepseek-v4-pro`).
Available on NIM free tier. 4 API keys for rate limit rotation.

## Integration

This project produces fine-tuned adapters consumed by the Godspeed coding agent.
The Godspeed driver catalog registers ZAYA1 under `openai/zaya1-8b`.

## Benchmark Targets

| Benchmark | Current ZAYA1-8B | Target (SFT+GRPO) |
|-----------|-----------------|-------------------|
| BFCL-v4 | 39.22 | 50+ |
| τ² (agentic) | 43.12 | 65–75 |
| AIME '26 | 89.1 | ≥84 (hard floor) |
| LiveCodeBench-v6 | 65.8 | ≥60 |

## License

Apache 2.0 — matches the ZAYA1-8B upstream license.
