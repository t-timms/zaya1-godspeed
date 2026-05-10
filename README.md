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
│   ├── test_peft.py          # PEFT LoRA attach + gradient flow test
│   └── serve.py              # vLLM server launcher for ZAYA1-8B
├── configs/
│   └── lora_tool_call.yaml   # QLoRA config for tool-calling fine-tune
├── data/
│   └── generate.py           # Convert Godspeed sessions to training data
├── COMPATIBILITY.md           # Architecture analysis & PEFT gate results
└── README.md
```

## Integration

This project produces fine-tuned adapters consumed by the Godspeed coding agent.
The Godspeed driver catalog registers ZAYA1 under `openai/zaya1-8b`.

## License

Apache 2.0 — matches the ZAYA1-8B upstream license.
