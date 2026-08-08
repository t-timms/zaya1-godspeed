# Model Selection — Why ZAYA1-8B for a 16 GB Local Coding Agent

> Original survey: 2026-06-02. **Refreshed: 2026-08-08.**
> Hardware: RTX 5070 Ti, 16 GB, consumer Blackwell (SM120).
> End goal: drive the [Godspeed coding agent](https://github.com/t-timms/godspeed-coding-agent)
> locally after the agentic fine-tune lands.

## 2026-08-08 refresh — what the original survey missed

The 2026-06-02 pass omitted two model families that were already public, and
predated a third development. All sizes below are measured from the published
Hugging Face artifacts, not estimated.

| Candidate | 4-bit artifact | Size | Fits 16 GB? |
|---|---|---:|---|
| **Gemma 4 12B** (`google/gemma-4-12B-it-qat-w4a16-ct`) | official Google **QAT**, compressed-tensors | **10.26 GB** | ✅ ~5.7 GB headroom |
| **Gemma 4 26B-A4B** (`...-qat-q4_0-gguf`) | official QAT GGUF | **14.44 GB** (+1.19 mmproj) | ⚠️ loads, but ~1.5 GB left — not agent-usable |
| **Qwen3.6-35B-A3B** (`nvidia/...-NVFP4`) | NVIDIA ModelOpt NVFP4 | **23.4 GB** (3 shards) | ❌ |
| **ZAYA1-8B** (this project) | compressed-tensors NVFP4 **W4A4** | 9.46 GB | ✅ |

Released 2026-04-15 (Qwen3.6) and 2026-04-02 (Gemma 4), both were available at
original survey time and should have been evaluated. The verdict does not change
— neither displaces ZAYA1-8B on a 16 GB card — but the omission is recorded here
rather than quietly corrected.

> A widely-shared blog claims Qwen3.6-35B-A3B fits a 5070 Ti at 10.88 GB. That
> figure is for APEX Nano, a different quantization. The NVFP4 export is 23.4 GB.

### The material change: vendors now ship their own quants

Since the original survey, first-party 4-bit releases have become the norm:

- **Zyphra** shipped `ZAYA1-8B-MXFP4-Experts` (5.85 GB) and `ZAYA1-8B-FP8-Experts`
  on 2026-07-02 — for the very model this project quantizes, and smaller.
- **Google** shipped QAT `w4a16` compressed-tensors across the Gemma 4 family.
- **NVIDIA** shipped NVFP4 for Qwen3.6-35B-A3B.

Post-training weight quantization is becoming a commodity. **But every one of
those first-party releases within a 16 GB budget is weights-only.** Verified from
config: Zyphra's MXFP4 has `ignore: "re:^(?!.*experts).*$"` and a `weights` block
with **no `input_activations`**; Google's is `w4a16` by name. NVIDIA's Qwen3.6
NVFP4 *is* W4A4, but at 23.4 GB it does not fit this card.

**Surviving niche: 4-bit activations inside 16 GB.** That is the claim this
project should lead with — not "4-bit weights," which is now table stakes.

### Revised triggers

1. **Want a strong general/multimodal local model today, no quantization work:**
   `google/gemma-4-12B-it-qat-w4a16-ct` — 10.26 GB, official QAT, vision+audio,
   256K context. Strictly better than hand-quantizing something comparable.
2. **VRAM upgrade to 24 GB+:** revisit Qwen3.6-35B-A3B NVFP4 (23.4 GB) and
   Gemma 4 26B-A4B with real KV headroom.
3. **The W4A4 thesis stays** — no first-party release covers 4-bit activations
   in this VRAM budget.

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
| Compute capability | SM120 (`gpu_capability [12, 0]`) | SM120 — **same arch; VRAM budget is the difference, not SM120 support** |
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
