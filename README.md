# zaya1-godspeed

Fine-tuning Zyphra's ZAYA1-8B for structured multi-turn tool calling with the
[Godspeed coding agent](https://github.com/omnipotence-eth/godspeed-coding-agent).

ZAYA1-8B punches well above its weight on math and coding benchmarks
(760M active / 8.4B total params, MoE). Per its own [technical report](https://arxiv.org/abs/2605.05365)
(May 2026), Zyphra **deliberately skipped** the multi-turn agentic RL stage —
the agentic gap is a training omission, not a capability ceiling. This project
completes ZAYA1-8B by adding that missing stage.

## Quick Start

```bash
uv sync
python scripts/test_peft.py   # verify PEFT compatibility
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
