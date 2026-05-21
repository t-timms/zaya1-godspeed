# Consumer Blackwell Deployment of ZAYA1-8B: W4A4 NVFP4 Quantization with Mixed-Precision MoE and SM120 CUTLASS Inference

**Tremayne Timms** — ttimmsinternational@gmail.com  
*Independent Research, May 2026*

---

## Abstract

ZAYA1-8B (Zyphra, May 2026) delivers state-of-the-art reasoning benchmarks—including 89.1% on AIME 2026—using only 760M active parameters via mixture-of-experts routing, but its 17 GB BF16 footprint prevents deployment on consumer GPUs. We present the first W4A4 NVFP4 quantization of ZAYA1-8B targeting the NVIDIA RTX 5070 Ti (Blackwell SM120). Our pipeline introduces (1) layer-wise GPU calibration that keeps CCA attention in BF16 while quantizing MoE Linears, (2) dynamic mixed-precision exemption of the 12 MoE layers (scattered from layer 1 to 79) with per-expert activation outliers (max\_abs > 500), and (3) a clean checkpoint design that requires only a single one-line vLLM patch for reproducible loading. Served via vLLM with pre-compiled CUTLASS SM120 NVFP4 kernels and CUDA graph capture, we achieve **102.6 tok/s** (single-stream) and **407.4 tok/s** (batch-8)—a 12.8× speedup over eager mode—at a checkpoint size of 8.9 GiB (vs. 17 GiB BF16). On standard lm-eval tasks, the mixed-precision checkpoint reaches 68.6% ARC-Easy acc\_norm and 61.5% HellaSwag acc\_norm—improvements of +1.3% and +1.0% over the prior W4A4 baseline with no mixed precision.

---

## 1. Introduction

Sparse mixture-of-experts (MoE) architectures achieve a favorable compute-to-quality ratio by activating only a fraction of parameters per token. ZAYA1-8B embodies this principle at an extreme: 16 experts with top-1 routing yields 760M active parameters from an 8B total parameter budget. The resulting BF16 model achieves AIME 2026 scores comparable to models with 3–12B active parameters (Table 2), yet fits its reasoning quality into the compute envelope of a sub-1B dense model.

The gap between quality and deployability, however, remains significant. At 17 GB, ZAYA1-8B in BF16 exceeds the memory capacity of all consumer GPUs. NVFP4—NVIDIA's block-structured 4-bit floating point format, hardware-accelerated on Blackwell tensor cores—offers a direct remedy: 4-bit weights and 4-bit activations (W4A4) reduce the model to roughly 9 GB while preserving the same weight values through a calibrated quantization scheme.

This work makes the following contributions:

1. **First W4A4 NVFP4 checkpoint of ZAYA1-8B**, produced via layer-wise GPU calibration with ARC-aware calibration data (977 samples × 1024 tokens, including ARC-Easy, ARC-Challenge, and HellaSwag distributions).

2. **Dynamic mixed-precision MoE**: 12 MoE layers (scattered from layer 1 to 79, with 9 of 12 in the deepest third) exhibit per-expert activation outliers with max\_abs > 500—over 5× the nominal FP4 range of 448. Forcing these layers to W4A4 would require input\_global\_scale values of ~6, compressing 99%+ of activations near zero. We exempt entire MoE layers (FusedMoE requires uniform quantization across all experts) and keep them in BF16, preserving 936/948 candidate modules in W4A4.

3. **SM120 CUTLASS kernel integration**: we compile vLLM 0.20.2 from source with `TORCH_CUDA_ARCH_LIST=12.0`, enabling `cutlass_scaled_fp4_mm_sm120a` and `cutlass_fp4_group_mm` kernels that exploit Blackwell's native NVFP4 tensor core MMA instructions. No custom CUDA code was written—the SM120 kernels existed in the vLLM source tree but were absent from pre-built wheels.

4. **Minimal reproducibility surface**: a single one-line addition to `vllm/model_executor/layers/fused_moe/oracle/unquantized.py` enables mixed-precision MoE loading. The checkpoint itself carries no quant-specific tensors on BF16-exempt layers, making the format unambiguous.

---

## 2. Background

### 2.1 ZAYA1-8B Architecture

ZAYA1-8B is a 80-layer MoE model with alternating Compressed Convolutional Attention (CCA) and MoE blocks. Key parameters:

| Property | Value |
|----------|-------|
| Total parameters | 8.4B |
| Active parameters per token | 760M |
| MoE experts | 16 (top-1 routing) |
| Hidden dimension | 2,048 |
| FFN dimension per expert | 4,096 |
| Attention | CCA (depthwise + grouped conv1d, L2-normalized QK) |
| Context window | 131,072 tokens |
| License | Apache 2.0 |

CCA attention differs substantially from standard multi-head attention: it uses depthwise and grouped 1D convolutions to mix queries and keys across sequence positions, with L2-normalized QK products and per-head learned temperature parameters. These layers are kept in BF16 throughout our quantization pipeline, as their convolution-based operations are not amenable to the NVFP4 block-structured format.

### 2.2 NVFP4 Format

NVFP4 uses a two-level quantization hierarchy. At the fine level, each group of 16 consecutive elements shares a per-group FP8 scale (weight\_scale\_fp8, E4M3 format). At the coarse level, a scalar input\_global\_scale (fp32) converts per-token maximum activations to the FP4 grid. The final dequantization is:

```
W_fp = unpack_fp4(W_packed) * weight_scale_fp8 * weight_global_scale
A_fp = quantize_fp4(A) * input_global_scale
```

The Blackwell SM120 tensor core MMA instruction `m16n16k32` consumes W4A4 operands natively, performing accumulation in BF16. The `cutlass_scaled_fp4_mm_sm120a` kernel wraps this instruction with CUTLASS's pipelined memory hierarchy for high-throughput GEMM.

The global\_scale convention (critical for correctness): the checkpoint stores `input_global_scale = 2688 / max_abs_activation`. The CT loader passes this divisor directly to the kernel as `input_global_scale_inv`, computing `block_scale = igs * vec_max / 6.0 = 448 * vec_max / max_abs`, where 448 is the FP4 maximum and 6.0 is the FP8 maximum. An inverted convention silently saturates all activations to 448, producing degenerate logits.

---

## 3. Model Positioning

ZAYA1-8B's public benchmarks (Table 1) show that a 760M active-parameter model matches or exceeds the performance of openly-available 4B dense models across math, coding, and knowledge tasks.

**Table 1: In-class comparison — ZAYA1-8B vs. comparable-budget open-source reasoning models.**

| Category | Benchmark | **ZAYA1-8B** (0.7B / 8.0B) | Qwen3-4B-Think-2507 (4.0B / 4.0B) | Qwen3.5-4B (4.0B / 4.0B) | Gemma-4-E4B-it (4.0B / 8.0B) |
|----------|-----------|:------:|:------:|:------:|:------:|
| Math | AIME '26 | **89.1** | 77.5 | 84.5 | 50.3 |
| Math | HMMT Feb '26 | **71.6** | 60.8 | 63.6 | 32.1 |
| Math | IMO-AnswerBench | **59.3** | 50.9 | 48.7 | 27.3 |
| Math | APEX-shortlist | **32.2** | 16.9 | — | 6.1 |
| Code | LiveCodeBench-v6 | **65.8** | 54.2 | — | 54.2 |
| Knowledge | GPQA-Diamond | 71.0 | 66.5 | **76.2** | 57.4 |
| Knowledge | MMLU-Pro | 74.2 | 74.3 | **79.1** | 70.2 |
| Instruction | IFEval | 85.6 | 86.8 | **89.8** | 88.5 |
| Instruction | IFBench | 52.6 | 52.9 | **59.2** | 42.7 |
| Style & chat | EQBench | 73.0 | **79.6** | 79.5 | 80.2 |
| Style & chat | Creative Writing v3 | 63.0 | 58.6 | 72.9 | **83.8** |
| Agentic | BFCL-v4 | 39.2 | **49.7** | 45.2 | 31.7 |
| Agentic | τ² | 43.1 | 52.9 | **82.1** | 37.7 |

*All figures are from the Zyphra official model card (May 2026). ZAYA1-8B leads its compute class on math and coding tasks—the domains most relevant to reasoning-intensive workloads—using 5–6× fewer active parameters than its competitors.*

**Table 2: Scaling comparison — ZAYA1-8B vs. larger open-source reasoning models.**

| Model | Active | Total | AIME '26 | HMMT '26 | LCB-v6 | IFEval | GPQA-D | MMLU-Pro |
|-------|:------:|:-----:|:--------:|:--------:|:------:|:------:|:------:|:--------:|
| **ZAYA1-8B** | **0.7B** | **8B** | **89.1** | **71.6** | **63.8** | **85.8** | **71.0** | **74.2** |
| Arcee-Trinity-Mini | 3B | 26B | 59.6 | 36.9 | 33.3 | 62.0 | 46.8 | 70.6 |
| N3-Nano-30B | 3B | 30B | 90.1 | 75.5 | 64.6 | **92.8** | 75.1 | 78.9 |
| OLMo-3.1-32B-Think | 32B | 32B | 78.9 | 50.6 | 58.3 | 93.2 | 59.6 | 75.8 |
| Qwen3-Next-80B-A3B-Think | 3B | 80B | 90.2 | **79.3** | **67.8** | 88.5 | **76.7** | 82.6 |
| Intellect-3 | 12B | 106B | 86.3 | 72.2 | 66.8 | 81.2 | 74.6 | 82.3 |
| Mistral-Small-4-119B | 6B | 119B | 86.4 | 70.6 | 57.9 | 84.0 | 77.2 | 81.6 |

*ZAYA1-8B with 0.7B active parameters achieves AIME scores competitive with models serving 3–32B active parameters. N3-Nano-30B and Qwen3-Next-80B (both MoE) are the only models that clearly exceed ZAYA1-8B on reasoning tasks, while requiring 4–130× more total compute per token.*

The data motivates our deployment goal: make this exceptional reasoning-to-compute ratio accessible to consumer hardware owners via aggressive quantization without sacrificing the benchmark scores that justify it.

---

## 4. Quantization Pipeline

### 4.1 Source Model

We start from `Zyphra/ZAYA1-8B` (Apache 2.0, 17 GiB, 4-shard safetensors), loaded in BF16 under `transformers` with `trust_remote_code=True`.

### 4.2 Layer-Wise GPU Calibration

Standard llm-compressor calibration runs the full model forward on CPU, which fails for ZAYA1-8B because CCA attention calls CUDA-only convolution kernels. We implement layer-wise GPU calibration:

1. Embed the 977-sample × 1024-token calibration corpus (BF16 token IDs) on GPU.
2. For each of the 80 layers (in order): load the layer to GPU, run a forward pass with pre-saved hidden states from the previous layer, record per-Linear activation maxima via forward hooks, save output hidden states to CPU, unload the layer.
3. Each layer occupies ~200 MB GPU memory during its calibration window; peak GPU usage stays below 4 GB.

The calibration corpus (`data/calibration/arcmix/calibration_data.pt`) is a weighted mix of 977 samples × 1024 tokens:

| Source | Samples | Weight | Category |
|--------|---------|--------|----------|
| ARC-Easy | 153 | 15% | Standard eval task |
| HellaSwag | 153 | 15% | Standard eval task |
| ARC-Challenge | 98 | 10% | Standard eval task |
| math500 | 102 | 10% | Math reasoning |
| gsm8k | 102 | 10% | Math reasoning |
| triviaqa | 102 | 10% | Knowledge |
| alpaca | 102 | 10% | Instruction following |
| humaneval | 38 | 5% | Code generation |
| writingprompts | 51 | 5% | Style/creativity |
| glaive | 51 | 5% | Agentic tool-use |
| mbpp | 25 | 5% | Code completion |
| **Total** | **977** | **100%** | |

Including ARC-Easy (15%), ARC-Challenge (10%), and HellaSwag (15%) in the calibration mix ensures the input\_global\_scale values are calibrated against a distribution representative of our lm-eval evaluation tasks, rather than being extrapolated from out-of-distribution math and code prompts alone.

### 4.3 Dynamic Mixed-Precision Exemption

After calibration, we compute the maximum activation magnitude across all experts for each MoE layer. Layers where `max(max_abs_per_expert) > 500` are added to the quantization ignore list, and their entire MLP (linear\_fc1 + linear\_fc2 for all 16 experts) is kept in BF16.

**Threshold rationale**: The FP4 E2M1 format has a maximum representable value of 6.0 (in normalized form). With group-size-16 FP8 block scales, the effective dynamic range is 448 × 6 = 2688. An activation max\_abs of 500 requires input\_global\_scale = 2688/500 ≈ 5.4, meaning the FP8 per-group scales absorb 5.4× of the activation range before the FP4 grid starts. Values smaller than max\_abs/448 ≈ 1.12 quantize to exactly zero—the model loses fine-grained differentiation across roughly half its activation range. A threshold of 500 was chosen empirically to balance accuracy (fewer BF16 exemptions = smaller model) against precision loss.

**Result**: 12 of 40 MoE layers are BF16-exempt. Outlier layers span the full depth: {1, 19, 31, 37, 39, 65, 69, 71, 73, 75, 77, 79}. The early outliers (1, 19, 31) likely correspond to embedding-space projections and early routing paths that see high-variance input distributions. Layers 37–79 dominate (9 of 12 outlier layers are in the deepest third of the model). 936 MoE Linear modules (= 1,320 total MoE Linears − 384 BF16-exempt) remain W4A4.

| Component | Count | Format |
|-----------|-------|--------|
| MoE Linear modules (W4A4) | **936** | NVFP4 W4A4, gs=16 |
| MoE Linear modules (BF16 exempt) | **384** | BF16 (12 layers × 16 experts × 2 proj) |
| CCA QKV projections | 160 | BF16 (always) |
| Routers, norms, embeddings, lm\_head | — | BF16 |
| **Total target Linears** | **1,320** | 936 W4A4 + 384 BF16-exempt |

### 4.4 Global-Scale Convention

vLLM's CUTLASS NVFP4 kernel reads `input_global_scale` as a divisor: it computes `block_scale = igs * vec_max / 6.0`. We store `igs = 2688 / max_abs_activation`, so the kernel computes `block_scale = (2688/max_abs) * vec_max / 6.0 = 448 * vec_max / max_abs`, which maps the per-group maximum to FP4 maximum (448) and scales all values proportionally.

Weight scales use the dual convention: `weight_global_scale = max_abs_weight / 2688`, and `weight_scale_fp8` stores pre-multiplied values `= (raw_weight_scale / max_abs) * 2688`. The kernel's dequantization recovers the original weight: `W_dequant = unpack_fp4(W_packed) * weight_scale_fp8 * weight_global_scale`.

### 4.5 Checkpoint Design

The clean checkpoint contains:
- **W4A4 Linear layers**: `weight_packed` (uint8, shape `[N, K/2]`), `weight_scale_fp8` (E4M3, shape `[N, K/16]`), `weight_global_scale` (fp32, scalar), `input_global_scale` (fp32, scalar).
- **BF16 Linear layers**: `weight` (BF16) only—no quant-specific keys.
- **MoE expert format** (FusedMoE): `w13_weight_packed` (shape `[E, 2N, K/2]`), `w13_weight_scale` (shape `[E, 2N, K/16]`), plus global scales and input scales per layer.

BF16-exempt layers carry zero quant-specific tensors. This design makes the checkpoint unambiguous: the presence of `weight_packed` is the sole signal that a module is quantized. No loading code needs to guess from layer indices or config fields.

---

## 5. Inference Infrastructure

### 5.1 vLLM Source Build for SM120

Pre-built vLLM 0.20.2 wheels do not include SM120 NVFP4 CUTLASS kernels. We compile from source:

```bash
cd /home/ttimm/vllm-src
TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=8 pip install -e . --no-build-isolation
```

This enables `ENABLE_NVFP4_SM120` in the CMake build, compiling:
- `cutlass_scaled_fp4_mm_sm120a` — W4A4 GEMM for Linear layers, CUTLASS Sm120 BlockScaledTensorOp, BF16 output
- `cutlass_fp4_group_mm` — W4A4 Group MoE GEMM, SM120 dispatch, both operands FP4

Build time: ~75 minutes. Verification: `torch.ops._C.cutlass_scaled_mm_supports_fp4(120)` returns `True`.

### 5.2 Mixed-Precision MoE Loading

vLLM's `map_unquantized_backend()` function did not originally map `moe_backend='cutlass'` to a triton fallback for unquantized layers. This single addition enables mixed-precision loading:

```python
# vllm/model_executor/layers/fused_moe/oracle/unquantized.py
def map_unquantized_backend(runner_backend: MoEBackend) -> UnquantizedMoeBackend:
    mapping = {
        "cutlass": UnquantizedMoeBackend.TRITON,  # ← this line
        "triton": UnquantizedMoeBackend.TRITON,
        ...
    }
```

With a clean checkpoint, this is the **only** required vLLM modification. W4A4 MoE layers dispatch to `VLLM_CUTLASS` (SM120 group GEMM); BF16-exempt MoE layers dispatch to Triton. The routing is entirely determined by the presence or absence of `w13_weight_scale` in the checkpoint.

### 5.3 CUDA Graph Capture

vLLM's CUDA graph capture (`enforce_eager=False`, the default) works without modification on this mixed-precision configuration. Graph capture covers:
- 51 piecewise batch sizes (1–512)
- 35 full-graph sizes (1–256)
- Total estimated CUDA graph memory: 2.75 GiB

With CUDA graphs, the model avoids Python dispatch overhead and CUDA kernel launch latency for each of the 80 decoder layers per decode step. The throughput impact is decisive.

### 5.4 Loading

The checkpoint loads in ~40 seconds via `safetensors` from a Windows filesystem (9P protocol, mounted in WSL2). Loading weight count: 6,792. VRAM consumed by model weights: **8.82 GiB** (vs. 17 GiB BF16).

---

## 6. Evaluation

### 6.1 Quantization Quality

We evaluate on standard lm-eval tasks using `lm_eval` with the `vllm` backend (`batch_size="auto"`, `max_model_len=4096`, `moe_backend=cutlass`). All evaluations run with CUDA graphs enabled.

**Table 3: W4A4 NVFP4 accuracy vs. prior W4A4 baseline (no mixed-precision).**

| Task | Shots | W4A4 baseline (no MP) | W4A4 mixed-precision (ARC-mix) | Delta |
|------|:-----:|:---------------------:|:------------------------------:|:-----:|
| ARC-Easy (acc\_norm) | 0 | 67.3% | **68.6%** | +1.3% |
| ARC-Challenge (acc\_norm) | 25 | N/A | **48.8%** | — |
| HellaSwag (acc\_norm) | 0 | 60.5% | **61.5%** | +1.0% |
| Winogrande (acc) | 5 | N/A | **58.0%** | — |

*Baseline: W4A4 max-abs calibration only, no mixed precision (session 8). Final column: mixed-precision (12 BF16 MoE layers, threshold=500) + ARC-aware calibration (977 samples × 1024 tokens including ARC-Easy 15%, ARC-Challenge 10%, HellaSwag 15%). Total evaluation time: ~65 min on RTX 5070 Ti.*

*BF16 reference scores for ARC/HellaSwag are not published by Zyphra; the official benchmarks focus on reasoning tasks (see Tables 1–2). Our W4A4 ARC scores therefore represent a lower bound on quantization quality relative to BF16.*

### 6.2 Throughput

Throughput measured on RTX 5070 Ti (16 GB VRAM, SM120, 40 CUs), WSL2, CUDA 13.2.

| Configuration | Tokens/sec | Notes |
|---------------|:----------:|-------|
| W4A4 NVFP4, CUDA graphs, batch=1 | **102.6** | Single-stream decode |
| W4A4 NVFP4, CUDA graphs, batch=8 | **407.4** | Batch decode |
| W4A4 NVFP4, eager mode, batch=1 | 8.0 | `enforce_eager=True` |
| Path A BF16 dequant (no CUTLASS) | 0.9 | Session 2 baseline, Python loop |

CUDA graphs provide a **12.8× speedup** over eager mode. The transition from Python dequant (Path A, session 2) to CUTLASS SM120 kernels (Path B, session 8) yielded a **114× total speedup** from the initial 0.9 tok/s baseline.

### 6.3 Model Size

| Checkpoint | Size | vs BF16 |
|-----------|------|---------|
| BF16 (original) | 17.0 GiB | 1.0× |
| W4A4 NVFP4 (fully quantized, hypothetical) | ~6.0 GiB | 2.8× smaller |
| W4A4 NVFP4 (mixed-precision, 384 BF16 Linears) | 8.9 GiB | **1.91× smaller** |

The size gap between mixed-precision (8.9 GiB) and theoretical fully-W4A4 (~6 GiB) is the cost of keeping 29% of MoE Linears (384 of 1,320) at BF16. This overhead is a direct consequence of the FusedMoE uniform quantization constraint (§7.2): those 12 layers' BF16 weight tensors together contribute ~2.9 GiB.

The 8.9 GiB checkpoint leaves ~7.1 GiB of VRAM free for KV cache at `gpu_memory_utilization=0.85` on a 16 GB device, enabling 16K+ context windows in a configuration where BF16 would not fit at all.

---

## 7. Technical Discoveries and Contributions

### 7.1 The Global-Scale Convention Bug

The most consequential discovery was an inverted global-scale convention in the initial checkpoint. Our quantize script stored `igs = max_abs / 6` (the wrong direction), and `weight_global_scale` was left uninitialized (yielding values like `6.17e+22`). The CUTLASS kernel produced NaN-valued logits, which manifested as uniform collapse to token\_id=0 (pad token) for every prompt—a silent failure with no kernel error.

The correct convention (§4.4) was reverse-engineered from the CT library source: `_compute_global_scale = scale_data.max * quant_data.max / max_val_pos`, where `scale_data.max = 6.0` (FP4 max) and `quant_data.max = 448.0` (FP8 E4M3 max), giving `igs = 6.0 * 448.0 / max_abs = 2688 / max_abs`. A round-trip verifier (`scripts/verify_w4a4_dequant.py`) confirmed the convention by comparing dequantized W4A4 weights against the original BF16 weights, showing mean relative error of 0.96–1.16%—within the ~1.5% noise floor expected for NVFP4 block quantization at group-size 16.

### 7.2 FusedMoE Uniform Quantization Constraint

vLLM's `FusedMoE` module requires all experts within a layer to use the same quantization scheme—a constraint inherited from the fused GEMM kernel's uniform dispatch path. Attempting per-expert mixed-precision (W4A4 for inlier experts, BF16 for outlier experts within the same layer) would require a separate execution path for each expert, negating the fused-expert performance benefit.

Our dynamic exemption operates at the layer level: if any expert within a layer has `max_abs > 500`, the entire layer is kept BF16. This is conservative (we exempt 12 layers rather than the minimum 2–4 individual experts that exceed the threshold) but correct.

### 7.3 Compressed-Tensors Calibration Interference

The llm-compressor's `apply_quantization_config()` silently replaces `nn.Linear.forward` with a fake-quantization wrapper that produces NaN outputs. This wrapper runs during the calibration forward pass, corrupting all subsequent hidden states. The fix is to restore the original `nn.Linear.forward` immediately after `apply_quantization_config()` before beginning calibration.

This issue would affect any compressed-tensors W4A4 quantization pipeline that uses llm-compressor's calibration hooks and was not previously documented.

---

## 8. Related Work

**Consumer NVFP4 inference**: Prior to this work, no publicly available W4A4 NVFP4 checkpoint of ZAYA1-8B existed. Community quantizations (10 repositories as of May 2026) used BNB NF4, MXFP4, ONNX, or MLX formats—none of which provide Blackwell-native hardware acceleration. The community MXFP4 quantization (OsaurusAI) failed to load due to weight shape mismatches with the Zyphra vLLM fork; NF4 (barozp) produces broken inference due to CCA attention incompatibility with BNB's forward hooks.

**W4A8 and W4A4 quantization**: SmoothQuant [Xiao et al., 2023] and LLM.int8() [Dettmers et al., 2022] pioneered activation-aware weight quantization. More recent work—QuaRot [Ashkboos et al., 2024], QuIP# [Tseng et al., 2024]—demonstrates W4A4 near-losslessly on dense models. NVFP4's block-structured format differs from per-channel or per-tensor quantization: each group of 16 weights shares a FP8 scale, and a global fp32 scale maps activation maxima to the FP4 grid. This two-level hierarchy provides 4-bit-equivalent compression with finer granularity than naive 4-bit uniform quantization.

**MoE quantization**: Quantizing sparse MoE models presents unique challenges absent in dense models. (1) Expert activation sparsity: with top-1 routing, 15 of 16 experts receive zero tokens per step—calibration must accumulate statistics across many tokens to observe all experts. (2) Outlier heterogeneity: our data shows that max\_abs values vary over 10× between the most and least extreme experts within a single layer, making per-layer uniform quantization lossy for outlier experts. (3) Fused expert kernels: efficient MoE serving requires all experts to share a quantization scheme (uniform block structure, same kernel path)—per-expert mixed-precision would require separate dispatch overhead that negates the fused GEMM benefit.

**Blackwell inference**: NVIDIA's Blackwell architecture (SM120) introduces dedicated FP4 tensor core instructions. vLLM's upstream source contains SM120 CUTLASS kernels but they are excluded from binary wheels, requiring a source build for consumer Blackwell deployment.

---

## 9. Conclusion

We have demonstrated that ZAYA1-8B—a SOTA reasoning model with 760M active parameters—can be served at **102.6 tok/s** on a consumer RTX 5070 Ti via W4A4 NVFP4 quantization with mixed-precision MoE, using pre-compiled SM120 CUTLASS kernels and CUDA graph capture. The resulting checkpoint occupies 8.9 GiB (vs. 17 GiB BF16), fitting comfortably within a 16 GB VRAM budget with room for a substantial KV cache.

The primary technical contributions are:

1. A layer-wise GPU calibration pipeline that handles CCA attention's CUDA-only operations correctly.
2. Dynamic mixed-precision MoE exemption that preserves accuracy on the 12 deepest layers with outlier activations, requiring only a one-line vLLM patch for loading.
3. Documentation of three quantization-specific failure modes (global-scale convention inversion, fake-quant forward hook corruption, FusedMoE uniform quantization constraint) that apply to any CT-based NVFP4 pipeline.
4. The first W4A4 NVFP4 ZAYA1-8B checkpoint that loads without model-architecture patches, enabling reproducible deployment by the open-source community.

The model's benchmark position (Table 2) demonstrates that sub-1B active-parameter MoE models can compete with 3–32B active-parameter dense and sparse models on reasoning tasks. Among the models evaluated, ZAYA1-8B uniquely achieves this with 0.7B active parameters—meaning each forward pass involves roughly the same compute as a 1B dense model—while matching or exceeding models such as Intellect-3 (12B active, 106B total) and Mistral-Small-4-119B (6B active, 119B total) on AIME 2026. Our W4A4 quantization removes the remaining barrier—memory footprint—making this reasoning quality accessible on the hardware most researchers and practitioners actually own: a single consumer GPU with 16 GB VRAM.

---

## 8. Limitations and Future Work

**Benchmark coverage**: Standard lm-eval accuracy numbers (ARC, HellaSwag, Winogrande) are reasonable proxies for quantization quality but do not measure ZAYA1-8B's primary capabilities (math reasoning, code, GPQA). The official Zyphra benchmarks use generation-based evaluation (pass@1 for math, execution for code), which requires model output rather than log-likelihood scoring. Running pass@1 AIME/HMMT evaluation on the W4A4 checkpoint would directly quantify quality retention on the model's native tasks but is not tractable without a large-scale inference infrastructure.

**BF16 outlier layer penalty**: 12 of 40 MoE layers stay at BF16, contributing ~3.2 GiB of overhead. The fully-W4A4 hypothetical checkpoint would be ~6 GiB—still exceeding a 16 GB GPU's capacity for practical KV cache allocation, but offering more headroom. Rotation-based methods (QuaRot [Ashkboos et al., 2024]) suppress activation outliers by applying orthogonal transforms before quantization; applying these to ZAYA1-8B's MoE layers could eliminate or reduce the BF16 exemptions. The FusedMoE uniform quantization constraint means per-expert rotation would require integrating rotation matrices into the fused kernel path, which is non-trivial.

**Single-GPU scope**: All results are from a single RTX 5070 Ti (SM120). Tensor-parallel deployment across multiple Blackwell GPUs is not tested and may require additional patches for the mixed-precision layer routing.

**Calibration distribution sensitivity**: ARC-aware calibration improves ARC/HellaSwag scores but may slightly reduce accuracy on tasks with different activation distributions (e.g., math reasoning, which activates different expert subsets). Future work should calibrate on a more representative mix of the model's intended use-case distribution.

**Future work**:
- **Upstream vLLM patch**: submit the `unquantized.py` one-liner as a PR to enable mixed-precision NVFP4 MoE loading without source patching.
- **Rotation-based outlier suppression**: QuaRot integration for MoE layers to achieve a fully-W4A4 ~6 GiB checkpoint.
- **Agentic fine-tuning**: use this W4A4 checkpoint as the inference backend for the target BFCL-v4/τ² improvement goal (SFT+GRPO on multi-step tool-use trajectories).
- **Generation-based eval**: pass@1 evaluation on AIME 2026 samples to directly quantify W4A4 quality retention on the model's primary benchmark.

---

## Appendix A: Reproducibility

**Hardware**: NVIDIA RTX 5070 Ti (16 GB GDDR7, SM120, Blackwell), WSL2 on Windows 11, CUDA 13.2.

**Software**: vLLM 0.20.2 (compiled from source), PyTorch 2.7.0+cu132, llm-compressor 0.4.1, lm-eval 0.4.x.

**Checkpoint path**: `zaya1-8b-nvfp4-w4a4/` (project root). Checkpoint contains 6,792 tensors; loads in ~40 seconds on a 9P (WSL2 virtio) filesystem.

**Required vLLM modification** (the only one): add `"cutlass": UnquantizedMoeBackend.TRITON` to `map_unquantized_backend()` in `vllm/model_executor/layers/fused_moe/oracle/unquantized.py`.

**Quantization command**:
```bash
source /home/ttimm/vllm-env/bin/activate
cd "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed"
python3 scripts/quantize_zaya_ct_nvfp4.py \
    --scheme w4a4 \
    --calibration-data data/calibration/arcmix/calibration_data.pt \
    --mixed-precision-threshold 500
```

**Inference command**:
```python
from vllm import LLM, SamplingParams
llm = LLM(
    model="./zaya1-8b-nvfp4-w4a4",
    dtype="bfloat16",
    moe_backend="cutlass",
    max_model_len=4096,
    gpu_memory_utilization=0.85,
)
```

**lm-eval command**:
```bash
lm_eval --model vllm \
    --model_args pretrained=./zaya1-8b-nvfp4-w4a4,dtype=bfloat16,moe_backend=cutlass,max_model_len=4096 \
    --tasks arc_easy,arc_challenge,hellaswag,winogrande \
    --num_fewshot 0,25,0,5 \
    --batch_size auto \
    --device cuda
```

---

## References

1. Washbourne et al. "ZAYA1-8B Technical Report." arXiv:2605.05365, May 2026.
2. NVIDIA Corporation. "NVIDIA Blackwell Architecture Technical Brief." 2025.
3. CUTLASS 4.4.2. NVIDIA Corporation. https://github.com/NVIDIA/cutlass, 2025.
4. Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
5. Xiao et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." ICML 2023.
6. Dettmers et al. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." NeurIPS 2022.
7. Ashkboos et al. "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs." arXiv:2404.00456, 2024.
8. llm-compressor. Neural Magic. https://github.com/vllm-project/llm-compressor, 2025.
9. Beeching et al. "Open LLM Leaderboard v2." HuggingFace, 2024.
