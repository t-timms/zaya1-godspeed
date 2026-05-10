# zaya1-godspeed

Fine-tuning Zyphra's ZAYA1-8B for structured tool calling with the
[Godspeed coding agent](https://github.com/omnipotence-eth/godspeed-coding-agent).

ZAYA1-8B punches well above its weight on math and coding benchmarks
(760M active / 8.4B total params, MoE). It was not trained for multi-step
tool calling — this project aims to fix that with targeted QLoRA fine-tuning.

## Quick Start

```bash
uv sync
python scripts/test_peft.py   # verify PEFT compatibility
```

## Project Structure

```
zaya1-godspeed/
├── scripts/
│   ├── train.py               # QLoRA fine-tuning with TRL SFTTrainer
│   ├── serve_zaya1.py          # vLLM server launcher (n-gram spec, tool-call support)
│   ├── serve.py                # vLLM server launcher for base model
│   ├── test_peft.py            # PEFT LoRA attach + gradient flow test
│   └── build_vllm_detached.sh  # Reliable Zyphra vLLM fork build (WSL)
├── configs/
│   └── lora_tool_call.yaml     # QLoRA config for tool-calling fine-tune
├── data/
│   └── generate.py             # Convert Godspeed sessions to ChatML training data
├── COMPATIBILITY.md             # Architecture analysis & PEFT gate results
├── ROADMAP.md                   # 7-phase project plan with status tracking
└── README.md
```

## Integration

This project produces fine-tuned adapters consumed by the Godspeed coding agent.
The Godspeed driver catalog registers ZAYA1 under `openai/zaya1-8b`.

## License

Apache 2.0 — matches the ZAYA1-8B upstream license.
