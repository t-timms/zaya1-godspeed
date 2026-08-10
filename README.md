# zaya1-godspeed

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![vLLM](https://img.shields.io/badge/vLLM-0.20.2%20source%20build-1E90FF?style=flat-square)](https://github.com/vllm-project/vllm)
[![CUTLASS](https://img.shields.io/badge/CUTLASS-4.4.2%20SM120-76B900?style=flat-square)](https://github.com/NVIDIA/cutlass)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)](./LICENSE)

**NVFP4 W4A4 inference for Zyphra's ZAYA1-8B on consumer Blackwell (RTX 5070 Ti, SM120)** —
4-bit weights *and* 4-bit activations running on native CUTLASS FP4 tensor-core kernels,
at **102.6 tok/s single-stream / 407.4 tok/s batch-8** from a 9.46 GB mixed-precision
checkpoint, within a 16 GB VRAM budget.

One other public NVFP4 W4A4 ZAYA1 checkpoint exists — built with NVIDIA ModelOpt
and validated on a 96 GB workstation Blackwell card, with no published accuracy or
throughput numbers ([MODEL_SELECTION.md](./MODEL_SELECTION.md) has the full
comparison). That card is also SM120, so SM120 support itself is not the
distinction here — the distinction is doing it with compressed-tensors inside a
16 GB **consumer** VRAM budget, with measured accuracy and throughput published.
Everything here was built and debugged on a single RTX 5070 Ti.

## Highlights

| Result | Detail |
|--------|--------|
| **102.6 tok/s** single / **407.4 tok/s** batch-8 | vLLM + CUDA graphs on RTX 5070 Ti (12.8× over eager mode) |
| **9.46 GB checkpoint** | 936 Linears in packed NVFP4 W4A4, 384 outlier-sensitive Linears kept BF16 (mixed precision) — the BF16 exemptions are why this is larger than a uniformly-quantized NVFP4 export |
| **Checkpoint verified healthy** | Budget-forced GPQA-Diamond rises monotonically with reasoning budget — 45.8% → 62.5% at a 12k-token think budget vs Zyphra's BF16 CoT 71.0%. At n=24 the confidence interval is wide (~±19 pts), so treat this as a health signal, not a parity claim |
| **First ZAYA1-8B GGUF** | 4.76 GB NVFP4 GGUF (4.52 bpw), built early May 2026 against a patched llama.cpp — community GGUFs appeared later via [llama.cpp PR #23112](https://github.com/ggml-org/llama.cpp/pull/23112) |
| **vLLM SM120 source build** | `TORCH_CUDA_ARCH_LIST=12.0` build enabling `cutlass_scaled_fp4_mm_sm120a` + FP4 group MoE GEMM — kernels that ship in vLLM source but not in wheels |

> **Base model revision note.** This checkpoint was quantized from the original
> 80-layer ZAYA1-8B config (`num_hidden_layers: 80`, `moe_router_topk`,
> `zaya_use_eda` / `zaya_use_mod`, transformers 4.57.1). In late June 2026 Zyphra
> refactored ZAYA1-8B into upstream-transformers form — `Zyphra/ZAYA1-8B` now
> reports `num_hidden_layers: 40` with `layer_types: hybrid` and
> `num_experts_per_tok`, and the original was moved to
> [`Zyphra/ZAYA1-8B-legacy`](https://huggingface.co/Zyphra/ZAYA1-8B-legacy).
> Core dimensions are unchanged (hidden 2048, 16 experts, top-1 routing, vocab
> 262272), so this reads as a re-expression of the same model rather than a new
> one — but **reproduce against `Zyphra/ZAYA1-8B-legacy`**, not current
> `ZAYA1-8B`, or the scripts here will hit an architecture mismatch.

## Why this is hard

ZAYA1-8B is an 80-layer MoE (760M active / 8.4B total) with Zyphra's CCA
(compressed convolutional attention) — no stock quantization path works:

- **No upstream llama.cpp support** — ZAYA1 landed in llama.cpp only via
  [PR #23112](https://github.com/ggml-org/llama.cpp/pull/23112); for the W4A4
  serving path here vLLM is the only viable engine, and stock wheels don't
  compile the SM120 NVFP4 CUTLASS kernels.
- **W4A4 means calibrating activation scales**, not just weights. The
  compressed-tensors calibration path silently swaps `nn.Linear.forward` for a
  NaN-producing fake-quant wrapper; MoE expert sparsity feeds empty tensors to
  observer hooks; and the NVFP4 global-scale convention (`2688 / max_abs`,
  divisor form, with block scales pre-multiplied) is undocumented — getting it
  wrong produces silent pad-token collapse, not an error.
- **Evaluating a reasoning model at 4-bit is its own project.** Stock lm-eval
  harnesses scored the model below random because ZAYA never closes its
  `<think>` block within budget and answers in `\boxed{}` format. We built
  s1-style budget-forced harnesses (`scripts/eval_gpqa_budget_forced.py`,
  `scripts/eval_ifeval_budget_forced.py`) that cap the reasoning trace, inject
  the close, and score only the final answer — turning a fake "quantization
  damage" signal into a clean scaling curve.

| think budget | GPQA-Diamond (n=24, paired) | traces self-closing `</think>` |
|---|---|---|
| 2,500 | 45.8% | 1/24 |
| 5,000 | 45.8% | 2/24 |
| 12,000 | **62.5%** | 9/24 |

The gap to BF16 is the 16 GB context/reasoning-budget ceiling — not
quantization damage.

## Repo map

```
zaya1-godspeed/
├── scripts/
│   ├── quantize_zaya_ct_nvfp4.py       # NVFP4 W4A4 quantization + layer-wise activation calibration
│   ├── build_calibration_data.py       # Calibration mix (incl. ARC/HellaSwag phase-2 mix)
│   ├── fix_w4a4_global_scales.py       # Post-hoc global-scale convention repair
│   ├── verify_w4a4_dequant.py          # Round-trip dequant vs HF reference (≈1% rel. error)
│   ├── run_full_benchmarks.py          # lm-eval suite w/ CUDA graphs + chat template
│   ├── eval_gpqa_budget_forced.py      # s1-style budget forcing for GPQA-Diamond
│   ├── eval_ifeval_budget_forced.py    # Two-stage budget forcing + official IFEval checkers
│   ├── smoke_test_mixed_precision.py   # Load + coherence + speed gate
│   └── train.py / remap_to_zaya.py     # QLoRA SFT pipeline (agentic fine-tune phase)
├── RESEARCH.md      # Full debugging log — every bug, root cause, and fix
├── PAPER.md         # Write-up draft
├── ROADMAP.md       # Phase-by-phase status
└── COMPATIBILITY.md # Architecture analysis, PEFT gate, ZAYA XML format
```

`RESEARCH.md` is the most useful document in this repo if you're trying to run
NVFP4 W4A4 on your own SM120 card — it records the full chain of root causes:
the w13 gate/up shard-split trap, the tied NVFP4 `lm_head` → `embed_tokens`
bind, the bf16-only inference contract, the global-scale convention, and the
CUDA-graph memory-profiler flag that silently eats 3.5 GB on a 16 GB card.

## Reproduce

```bash
# 1. Build vLLM from source with SM120 NVFP4 kernels (WSL2, CUDA 13.x)
cd vllm-src && TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=8 pip install -e . --no-build-isolation

# 2. Quantize (layer-wise GPU calibration, ~10 min on 16 GB)
python scripts/build_calibration_data.py --arc-mix
python scripts/quantize_zaya_ct_nvfp4.py --scheme w4a4 --mixed-precision-threshold 500

# 3. Smoke test + benchmarks
python scripts/smoke_test_mixed_precision.py
python scripts/run_full_benchmarks.py
```

## Roadmap: the agentic fine-tune

The original goal stands: Zyphra's technical report
([arXiv 2605.05365](https://arxiv.org/abs/2605.05365)) deliberately skipped the
multi-turn agentic RL stage. With fast local W4A4 serving now solved, the next
phase distills multi-turn tool-calling trajectories from a teacher through the
[Godspeed coding agent](https://github.com/t-timms/godspeed-coding-agent)
harness into QLoRA SFT + GRPO — targeting BFCL-v4 and τ² gains on a model that
runs entirely on a 16 GB consumer card.

## License

Apache 2.0 — matches the ZAYA1-8B upstream license.
