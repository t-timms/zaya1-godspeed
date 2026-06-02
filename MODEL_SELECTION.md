# Model Selection — Why ZAYA1-8B for a 16 GB Local Coding Agent

> Survey date: 2026-06-02. Hardware: RTX 5070 Ti, 16 GB, consumer Blackwell (SM120).
> End goal: drive the [Godspeed coding agent](https://github.com/omnipotence-eth/godspeed-coding-agent)
> locally after the agentic fine-tune lands.

## The binding constraint

With vLLM on a 16 GB card, the CUDA-graph profiler alone reserves ~3.5 GiB (see
`gotcha_vllm_cuda_graph_profiling`), and you still need KV cache + activations.
The practical **weight budget is ~10 GB** if you want agent-usable context. That
single number filters the entire field — most purpose-built coding-agent models
are too large to run here at all.

## Has someone already hit our W4A4 quant goal?

**Yes — partially.** [`switzerchees/ZAYA1-8B-NVFP4`](https://huggingface.co/switzerchees/ZAYA1-8B-NVFP4)
(created 2026-05-19) is a genuine NVFP4 **W4A4** ZAYA1-8B (verified from its
`config.json`: both `weights` and `input_activations` at num_bits=4, group_size=16)
that loads on SM120 vLLM. The headline "does W4A4 NVFP4 ZAYA1 run on Blackwell"
question is answered — but with gaps that keep our work distinct:

| Dimension | switzerchees | This project |
|---|---|---|
| Quant toolchain | NVIDIA **ModelOpt** v0.44.0 | **compressed-tensors / llm-compressor** |
| vLLM | Zyphra prebuilt `zaya1-pr` branch | **Hand-built SM120 CUTLASS from source** |
| Hardware validated | RTX PRO 6000 (96 GB workstation Blackwell) | **RTX 5070 Ti, 16 GB consumer** |
| Accuracy eval | None — reposts model-card numbers | **Own lm-eval** (ARC, HellaSwag, GPQA/MMLU-Pro/IFEval) |
| Throughput reported | None | **102 tok/s single / 407 tok/s batch-8** (CUDA graphs) |
| Outlier handling | Not addressed | Mixed-precision outlier-layer exemption + SOAR/MR-GPTQ |

It's a useful **A/B baseline** to diff our checkpoint against, not a reason to stop.

## Is there a better base model for a local coding agent?

The trending purpose-built coders are excellent agents (native tool-calling, MoE
3B-active so fast) and already have Blackwell NVFP4 quants — but the people who
quantized them have 96–128 GB (DGX Spark / GB10, RTX PRO 6000). On a 5070 Ti they
don't fit.

| Model | Total params | ~NVFP4 weights | Fits 16 GB? | Agent out-of-box? |
|---|---|---|---|---|
| **ZAYA1-8B** (current) | 8.4B (760M active) | ~5–6 GB | ✅ large headroom | ❌ agentic stage skipped — *our fine-tune fills this* |
| Qwen3-Coder-30B-A3B-Instruct | 30.5B (3B active) | ~15–16 GB | ❌ no KV room | ✅ native tool-calling |
| Qwen3-Coder-Next | **79.7B** (3B active) | ~40 GB | ❌ not close | ✅ flagship agentic coder |
| Qwen3-Coder-REAP-25B-A3B (pruned) | 25B (3B active) | ~12–13 GB | ⚠️ very tight, tiny ctx | ✅ |
| Qwen2.5-Coder-14B (older dense) | 14B | ~7–8 GB | ✅ | ⚠️ decent, last-gen |

## Verdict

**Stay on ZAYA1-8B for this project.** It is the only model that is simultaneously
(a) strong on raw code/math for its size (LiveCodeBench 65.8, AIME 89.1),
(b) small enough to leave real KV headroom on 16 GB, and (c) the exact model whose
*missing* agentic stage our fine-tune supplies. Switching to an already-agentic
model would erase the project thesis ("complete what Zyphra skipped").

**Two triggers that would change the answer:**

1. **VRAM upgrade to 24 GB+ (or a no-fine-tune drop-in):** standardize on
   **Qwen3-Coder-30B-A3B-Instruct** — already agentic, MoE 3B-active, with many
   **compressed-tensors NVFP4** quants in our exact toolchain (`Firworks/`, `ig1/`,
   `GAlex535/`).
2. **Want a purpose-built agent that almost fits 16 GB:**
   **Qwen3-Coder-REAP-25B-A3B** (Cerebras-pruned, `Firworks/...-nvfp4` exists) —
   runnable only at short context, but the closest real agent that nearly fits.

## Portfolio action

When our ZAYA1 benchmarks pass, run **BFCL-v4 / τ²** on the fine-tuned ZAYA1 vs.
stock **Qwen3-Coder-30B-A3B** (via an inference provider — no local VRAM needed).
If an 8.4B (760M-active) fine-tune closes the agentic gap to a model 3.6× its size,
that is the headline result.

## Sources

- [Zyphra/ZAYA1-8B](https://huggingface.co/Zyphra/ZAYA1-8B)
- [switzerchees/ZAYA1-8B-NVFP4](https://huggingface.co/switzerchees/ZAYA1-8B-NVFP4)
- [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
- [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)
- [Firworks/Qwen3-Coder-REAP-25B-A3B-nvfp4](https://huggingface.co/Firworks/Qwen3-Coder-REAP-25B-A3B-nvfp4)
