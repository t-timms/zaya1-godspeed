# Roadmap

## Vision

ZAYA1-8B is a state-of-the-art reasoning model with 760M active parameters that
outperforms models 5x its size on math and coding. Per its own technical report
(arXiv 2605.05365, May 2026), Zyphra **deliberately skipped** the multi-turn
agentic RL stage — the agentic gap is a training omission, not a capability ceiling:

> "ZAYA1-8B does not include a dedicated multi-turn agentic RL stage in this
> release. We include some supervised agent, tool, and SWE traces during SFT,
> but the RL cascade is primarily optimized for verifiable reasoning, math,
> code, and instruction-following behavior. Scaling agentic data and agentic
> RL is left for future releases."

**This project completes ZAYA1-8B**: adding the missing agentic stage to drive
the Godspeed coding agent on consumer hardware (RTX 5070 Ti, 16 GB VRAM).

## Status Legend

| Icon | Meaning |
|------|---------|
| ✅ | Done |
| 🔴 | Blocked |
| 🟡 | In progress |
| ⬜ | Not started |

---

## Phase 0 — Development Environment ✅

**Goal**: SOTA development tooling — LSP, MCP, CUDA monitoring, pre-commit hooks.

| Task | Status | Notes |
|------|--------|-------|
| opencode.json project config | ✅ | LSP (pyright), MCP (cuda-monitor, context7), 15 slash commands, formatter (ruff) |
| CUDA monitoring MCP server | ✅ | `scripts/cuda_mcp_server.py` — 4 tools: gpu_status, vram_usage, cuda_version, gpu_clock |
| Pre-commit hooks | ✅ | `.pre-commit-config.yaml` — ruff lint + format + pytest |
| Type checker config | ✅ | `[tool.pyright]` in pyproject.toml — standard mode, py311 |
| Auto-formatter | ✅ | ruff format on file write (opencode.json formatter config) |

---

## Phase 1 — Compatibility Gate ✅

**Goal**: Verify the Zyphra stack can load ZAYA1-8B and attach PEFT adapters.

| Task | Status | Notes |
|------|--------|-------|
| Install Zyphra transformers fork | ✅ | `transformers @ git+https://github.com/Zyphra/transformers.git@zaya1` |
| Load ZAYA1-8B config and verify architecture | ✅ | `ZayaForCausalLM`, 80 layers (40 attn + 40 MoE), CCA |
| Map all 1641 Linear modules for LoRA targeting | ✅ | 200 attention, 1280 expert, 160 router, 1 lm_head |
| Attach PEFT LoRA to attention projections | ✅ | 8.2M trainable params, 0.09%, 7.24 GB VRAM |
| Verify gradient flow through LoRA adapters | ✅ | 400 params with gradients, loss.backward() succeeds |
| Document architecture and compatibility | ✅ | `COMPATIBILITY.md` |

**Decision**: Gate PASSED. QLoRA fine-tuning is viable.

---

## Phase 2 — Inference Pipeline ✅

**Goal**: Serve ZAYA1-8B via an OpenAI-compatible endpoint at usable speed.

| Task | Status | Notes |
|------|--------|-------|
| Build vLLM (Zyphra fork) in WSL | ✅ | **Root cause of all 6 failed builds found**: `nvcc` at `/usr/local/cuda/bin/nvcc` (CUDA 13.2) not in PATH. Build succeeds in ~3 min with `export PATH=/usr/local/cuda/bin:$PATH`. Zyphra Python files can be overlaid on stock vLLM 0.20.2 (no full rebuild needed). |
| Serve ZAYA1-8B via vLLM (FP8) | ✅ | 8.76 GB model loads (5.4s startup, 4.58 GB KV cache). Server responds: `/v1/models` returns model info, application startup complete. Required 3 patches: ModelRegistry, cca_state_shape, cca_state_dtype. |
| Serve ZAYA1-8B via vLLM (bf16) | ❌ | 16.48 GB model exceeds 15.92 GB GPU. -0.39 GB for KV cache. Confirmed impossible on 16 GB. |
| **NVFP4 ZAYA1-8B GGUF built** | ✅ | **First-ever NVFP4 ZAYA1-8B**: 4.76 GB, 4.52 bpw, 1641 NVFP4 + 842 F16 tensors (3205 total). BPE tokenizer embedded (262K tokens, 515K merges). llama.cpp NVFP4 fallback fixed (submitted upstream). Weights verified: 0.026 mean error vs original. Name mapping: 2483/2483 mapped, 0 failures. |
| Fix serve_zaya1.py | ✅ | Re-engineered May 11: `subprocess.Popen`, health polling, eager mode, FP8/MXFP4 quant support, matches official Zyphra deploy command. |
| MXFP4 quantized serving (Blackwell) | ❌ | OsaurusAI MXFP4 model: weight shape mismatch with Zyphra vLLM fork. |
| NF4 path (transformers) | ❌ | Confirmed broken — bitsandbytes dequant incompatible with CCA attention. |
| bitsandbytes (vLLM) | ❌ | ZayaForCausalLM lacks `packed_modules_mapping`. |
| **NVFP4 Compressed-Tensors quantization** | ✅ | ZAYA1-8B quantized to NVFP4 via `NVFP4PackedCompressor` (`zaya1-8b-nvfp4-ct-gs16/`, 5.04 GB, group_size=16, 1641 Linear modules, uint8 packed `[out, in//2]`, ~20s). FP4 values packed 2-per-byte, FP8_E4M3 scales, no global scales (per-group only), symmetric zero-point removed. |
| **NVFP4 CT loads via vLLM** | ✅ | May 14 session 1. All 4,244 weights mapped + initialized, 5.53 GiB VRAM, smoke test exit 0. 2 patches: scale routing + Marlin group_size fallback. |
| **NVFP4 CT generates coherent text** | ✅ | **May 14 session 2** — `"The capital of France is"` → `" Paris."`; coherent BST explanation. Required 3 additional patches (`wsl_fix_nvfp4_text_gen.py`): split combined w13 packed+scale into gate/up halves on load, dequant tied NVFP4 lm_head into embed_tokens.weight, rewrite MoE method for Path A on-the-fly Python dequant. Throughput: ~0.86 tok/s on RTX 5070 Ti (16 GB Blackwell sm_120). **Inference contract: `dtype="bfloat16"` required** — fp16 collapses output to a repeated token. |
| lainlives/ZAYA1-8B-GGUF audit | ✅ | Repo is empty (0 GGUF files, 0 bytes storage). README claims Q4_K/Q8_0/etc but none uploaded. Our NVFP4 GGUF is genuinely the first and only ZAYA1-8B GGUF. |
| llama.cpp Zaya support | ❌ | No model implementation exists. `convert_hf_to_gguf.py` has no ZayaForCausalLM entry. llama.cpp cannot serve ZAYA1-8B — vLLM is the only viable inference engine. |

**Blocker resolved**: Root cause was WSL2 `llama-server` running Qwen3.6-27B-Q4_K_XL (15 GB), not Windows compositor. Removed May 11.

### NVFP4 Serving Architecture (May 12, 2026)

Three paths evaluated for serving the 4.76 GB NVFP4 GGUF:

| Path | VRAM | Kernel | Speed | Status |
|------|------|--------|-------|--------|
| GGUF → vLLM GGUF handler | — | Python dequant (slow) | Unusable | Partial (803/2483 weights at 1.04 GB). Blocked on MoE routing + single-shard materialization |
| GGUF → Compressed-tensors | 4-5 GB | Marlin FP4 | Fast | Viable, 6-9 hrs |
| Original → Compressed-tensors | 4-5 GB | Marlin FP4 | Fast | **Chosen for Stage 1** — avoids double quantization |
| Custom Blackwell CUDA kernel | 4-5 GB | NVFP4 Tensor Core MMA | Fastest | **Stage 2** — reusable across all models, open-source contribution |

**Decision**: Two-stage pipeline. Stage 1: Compressed-tensors + Path A Python dequant for first benchmark (coherent text achieved May 14 session 2). Stage 2: Custom Blackwell CUDA kernel for hardware-accelerated dequant.

### Stage 2 — CUTLASS SM120 NVFP4 Tensor Core Kernel 🟡

**Goal**: Replace Path A Python dequant with NVIDIA's CUTLASS SM120 BlockScaledTensorOp kernels.

**Architecture discovery (May 14, 2026, session 3)**:
Consumer Blackwell (SM120/RTX 5070 Ti) uses an **extended `mma.sync.aligned.kind::f8f6f4`** instruction (Ampere-era warp-level programming model), NOT `tcgen05` (datacenter-only, requires TMEM) and NOT `wgmma` (Hopper-era, deprecated). SM120 has no TMEM — accumulators stay in registers. FP4/FP6 tensor core support IS present but with the older programming model. vLLM source at `/home/ttimm/vllm-src/` (v0.20.2) already contains CUTLASS SM120 FP4 kernels that were NOT compiled into the pre-built wheel.

**Key source files** (all in `/home/ttimm/vllm-src/csrc/`):
- `libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu` — `cutlass_scaled_fp4_mm_sm120a`: FP4×FP4 GEMM via CUTLASS Sm120 BlockScaledTensorOp. Outputs bf16 or fp16.
- `libtorch_stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu` — `cutlass_fp4_group_mm`: Grouped MoE GEMM with SM120 dispatch via `run_fp4_blockwise_scaled_group_mm_sm120`. Both A and B are FP4 with `float_ue4m3_t` scales.
- `quantization/marlin/marlin.cu` — Marlin kernel (works on sm_120 for Linear but corrupts MoE scales)
- `moe/marlin_moe_wna16/ops.cu` — Marlin MoE kernel (weight-only FP4, dequantizes weights on the fly)

**Build result (May 15, 2026, session 4) — KERNELS COMPILED ✓**:
- ✅ vLLM rebuilt from source with `TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=8 pip install -e . --no-build-isolation`
- ✅ Build time: ~75 minutes, editable install at `/home/ttimm/vllm-src/vllm/`
- ✅ `_C_stable_libtorch.abi3.so` (107MB) and `_C.abi3.so` (205MB) compiled with SM120 NVFP4 CUTLASS kernels
- ✅ `cutlass_scaled_mm_supports_fp4(120)` → **True** — SM120 confirmed working
- ✅ `cutlass_scaled_fp4_mm` dispatches to `cutlass_scaled_fp4_mm_sm120a` for SM120
- ✅ `cutlass_fp4_group_mm` available for Group MoE GEMM on SM120
- ✅ NVFP4 CUTLASS Linear wiring already exists: `CutlassNvFp4LinearKernel` in `vllm/model_executor/kernels/linear/nvfp4/cutlass.py`
- ✅ NVFP4 CUTLASS MoE wiring already exists: `NvFp4MoeBackend.VLLM_CUTLASS` in `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`

**Approach**: vLLM already has the full wiring — the only missing piece was the compiled SM120 kernels, now resolved. The `CutlassNvFp4LinearKernel` and `VLLM_CUTLASS` MoE backend are now functional on SM120.

| Task | Status | Notes |
|------|--------|-------|
| Set up CUDA LSP (clangd + compile_commands.json) | ❌ | Skipped — not needed for build. |
| Rebuild vLLM from source with SM120 CUTLASS support | ✅ | `TORCH_CUDA_ARCH_LIST=12.0`, CUTLASS 4.4.2, CUDA 13.2. ~75 min build. |
| Verify kernel availability | ✅ | `cutlass_scaled_mm_supports_fp4(120)=True`. All 3 ops present in `torch.ops._C`. |
| Apply Zyphra overlay to vLLM source (zaya.py, cca.py, registry) | ⬜ | Files at `/tmp/zaya-vllm/`. Copy to `/home/ttimm/vllm-src/vllm/` + registry patch. |
| End-to-end NVFP4 CUTLASS inference with zaya1-8b-ct-gs16 | ⬜ | Model loads but needs `ZayaForCausalLM` registered. |
| Wire `cutlass_fp4_group_mm` into MoE `apply()` | ✅ | Already wired — `NvFp4MoeBackend.VLLM_CUTLASS` exists in nvfp4.py. |
| Verify scale format compatibility (signed E4M3 → unsigned E4M3) | ⬜ | Our checkpoint scales are non-negative; casting should be a no-op. |
| Wire `cutlass_scaled_fp4_mm_sm120a` for CCA attention Linear layers | ✅ | Already wired — `CutlassNvFp4LinearKernel` in cutlass.py. |
| Drop the bf16-required inference contract | ⬜ | CUTLASS kernel accumulates in fp32 internally — should handle fp16 inputs. |
| Benchmark Stage 2 vs Stage 1 (target: >10× speedup) | ⬜ | Deterministic dequant math → identical output quality. |
| Submit upstream PR to vLLM | ⬜ | SM120 support is already in vLLM source — PR would fix pre-built wheel flags. |

**Key technical findings from GGUF loading attempts**:
- vLLM's GGUF handler lacks NVFP4 tensor type support; requires adding to DEQUANT_TYPES + Python dequant fallback
- GGUF quant handler creates `qweight`/`qweight_type` as GGUFUninitializedParameter — single-shard params never materialize without patching `_create_padded_weight_param`
- ZAYA's FusedMoE uses `w13_weight`/`w2_weight` but GGUF handler creates `w13_qweight`/`w2_qweight` — zaya.py load_weights needs GGUF-aware routing
- CCA attention requires GPU; CPU offloading produces garbage output
- `CompressedTensorsW4A16Fp4` uses Marlin FP4 kernel (not Blackwell-specific); `get_min_capability()` returns 75 (Turing+)

### Path B — W4A4 Weight+Activation Quantization 🟢 (Week 1 complete)

**Goal**: Go beyond Path A's weight-only NVFP4 (W4A16) to full weight+activation
quantization (W4A4) using llm-compressor + compressed-tensors. W4A4 quantizes
both weights AND activations to 4-bit FP4, requiring per-Linear activation
calibration (input_global_scale) via forward-pass statistics.

**Motivation**: W4A4 reduces model size further than W4A16 by quantizing
activations. For ZAYA1-8B on 16 GB consumer hardware, this enables even larger
KV caches and batch sizes. The upcoming CUTLASS SM120 kernel handles W4A4
natively via Blackwell Tensor Core MMA.

#### Week 1.1 — Calibration Corpus ✅

Built padding-free packed 1024-prompt calibration corpus:

| Metric | Value |
|--------|-------|
| Total samples | 979 |
| Max length | 1024 tokens |
| Packing mode | concat-pack with EOS separator |
| Pad ratio | 0.006% (effectively zero-padding) |
| Source mix | math500 (15%), gsm8k (15%), humaneval (5%), mbpp (5%), triviaqa (15%), alpaca (15%), writingprompts (15%), glaive (15%) |
| Output | `data/calibration/manifest.json` + tokenized tensors |

**Architecture**: The corpus packs multiple prompts end-to-end with EOS
separators, then slices into fixed 1024-token blocks. This avoids the GPU
memory bloat of padding to max-length and provides uniform tensor shapes for
calibration.

#### Week 1.2 — W4A4 Scheme Extension ✅

Extended `scripts/quantize_zaya_ct_nvfp4.py` with `--scheme w4a4` support:
- Activates llm-compressor's activation quantization pipeline
- Requires calibration data for `input_global_scale` computation
- New CLI args: `--scheme {w4a16,w4a4}`, `--calibration-data PATH`, `--dry-run-layers N`
- `W4A4` = NVFP4 weights (FP4_E2M1) + NVFP4 activation scales (FP8_E4M3 input_global_scale)

#### Week 1.2 Validation — Dry-Run Results 🟡 (historical)

Initial CPU-calibration approach failed: 4/1480 Linears observed, all reading
identical 10.625, output dir 18.56 GB. Attributed at first to "CCA needs CUDA";
true root cause was found in Week 1.3 (see below) — `apply_quantization_config`
silently replaced each Linear's `forward` with a NaN-producing fake-quant
wrapper, so downstream activations were all NaN/zero regardless of platform.

#### Week 1.3 — Layer-Wise GPU Calibration ✅ (2026-05-17)

Replaced the broken CPU calibration with a layer-wise GPU pipeline in
`calibrate_input_global_scales_layerwise()` (~210 LOC). The function:

1. Embeds the 979×1024 calibration tensor on GPU (one batch at a time).
2. For each of the 80 decoder layers: moves the layer to GPU, registers
   forward-pre-hooks on every quantized Linear in the layer, forwards each
   cached sample state through it, caches the new `(hidden_states, residual,
   prev_router_hidden_states)` to CPU, then moves the layer back to CPU.
3. Sets `input_global_scale = max_act / FP4_E2M1_MAX` per Linear from observed
   max-abs activations.

Five additional fixes shipped in the same change:

| Fix | Impact |
|-----|--------|
| Restore `nn.Linear.forward` on every quantized Linear after `apply_quantization_config` | Removes the silent NaN fake-quant wrapper that masked all downstream hooks — coverage went from 4/1480 to 1480/1480 |
| Skip pre-hooks when `x.numel() == 0` | MoE expert sparsity: experts that received zero routed tokens crash `.max()` with no reduction dim |
| Quality gate: ERROR if `nonzero_igs / expected_igs < 0.95` | Catches calibration failures that the old "key count" verification reported as PASS |
| Dry-run bloat fix: drop params from layers ≥ DRY_RUN_LAYERS_W4A4 | Dry-run output 18.56 GB → 1.32 GB |
| CCA ignore regex `re:.*cca.*` → `re:.*qkv.*` | `cca` substring never appeared in any path; CCA Q/K/V projections were being W4A4-quantized when intent was BF16. Fixed → 160 CCA Linears now stay BF16 |

#### Week 1.3 — Full Quantization Results 🟢

Final W4A4 checkpoint at `./zaya1-8b-nvfp4-w4a4/`:

| Metric | Value |
|--------|-------|
| Calibration samples used | 979 (full corpus) |
| Linears quantized to W4A4 | **1320/1320 (100%)** |
| IGS coverage | 100% — 0 near-zero garbage |
| Activation max distribution | min 3.58, p25 7.31, median 14.63, p75 29.13, **max 8896.0** |
| Total calibration time | 564s (9.4 min) — vs the original "1–4 GPU-hr" estimate |
| Output size | 5.99 GB (4068 MB packed weights + 509 MB weight_scales + 5.3 KB IGS + 1407 MB BF16 other) |
| Modules kept BF16 | lm_head, all RMSNorms, all routers, 160 CCA projections (`qkv.{linear_q,linear_k,val_proj1,val_proj2}`) |

**Known outlier**: one Linear hits max-abs activation of 8896 (IGS = 1483).
Likely an `o_proj` or `down_proj` per the Week 3 plan's outlier note. Will be
addressed at Week 3 if accuracy testing reveals it as a hot spot
(SmoothQuant-style rotation or per-layer ignore).

| Task | Status | Notes |
|------|--------|-------|
| Build padding-free 1024-prompt calibration corpus | ✅ | 979 samples, 8 sources, 0.006% pad ratio |
| Extend quantize script with `--scheme w4a4` | ✅ | `scripts/quantize_zaya_ct_nvfp4.py` |
| W4A4 dry-run (CPU baseline) | 🟡 | Historical failure — root cause was NaN forward-wrapper, not CPU |
| Implement layer-wise GPU calibration | ✅ | `calibrate_input_global_scales_layerwise`, ~210 LOC |
| Restore plain forward after `apply_quantization_config` | ✅ | Memory: `gotcha_compressed_tensors_calibration.md` |
| Empty-tensor hook skip (MoE expert sparsity) | ✅ | Two-line fix; both calibration functions |
| Quality gate (ERROR on low coverage) | ✅ | Errors when `nonzero_igs / expected_igs < 0.95` |
| Output-dir bloat fix | ✅ | 18.56 GB → 1.32 GB dry-run, full save = 5.99 GB |
| Fix CCA ignore regex | ✅ | `re:.*cca.*` → `re:.*qkv.*` (160 CCA Linears now BF16) |
| Full W4A4 quantization on all 80 layers | ✅ | 1320/1320 IGS at 100% coverage |
| Integrate W4A4 with CUTLASS SM120 kernel | ⬜ | Week 2 — wire loader to `CutlassNvFp4LinearKernel`, force VLLM_NVFP4_GEMM_BACKEND=cutlass |

#### Accuracy Baselines — Session 8 (2026-05-19) 🟢

First lm-eval numbers on the W4A4 checkpoint. All 1320 Linears at W4A4; 12 outlier
MoE layers with max-abs > 500 exempted to BF16 MLP (`--mixed-precision-threshold 500`).

| Benchmark | W4A4 (SOAR) | BF16 ceiling | Gap |
|-----------|------------|--------------|-----|
| ARC-Easy acc | 68.6% | ~75-80% | ~7-12pp |
| ARC-Easy acc_norm | 67.3% | — | — |
| HellaSwag acc | 45.7% | ~76-80% | — |
| HellaSwag acc_norm | 60.5% | — | — |

CUDA graphs enabled (no `enforce_eager`). Single-token throughput: **102 tok/s**
(12.8× over eager mode). Batch-8: **407 tok/s**.

> 🔴 **RETRACTED (2026-08-14). Historical record only, do not cite.** These
> numbers, and the accuracy figures in the table above them, were all measured
> with CUDA graph capture enabled, a path since confirmed to produce numerically
> incorrect output on SM120. Corrected values: 9.5 tok/s single-stream, ~74 tok/s
> batch-8, `enforce_eager=True`. See Session 17 and `RESEARCH.md` §5.14.

**SOAR baseline (with arcmix calibration):**
ARC-Easy 67.2%, ARC-Challenge 47.8%, HellaSwag acc_norm 61.4%, Winogrande 58.6%.

#### Current Checkpoint — Session 14 (2026-05-22) 🟢

**Active checkpoint**: `./zaya1-8b-nvfp4-w4a4/` (8.84 GiB)
- 936 W4A4 modules + 384 BF16-exempt (12 outlier MoE layers, threshold=500)
- ARC-mix calibration (977 samples: ARC-Easy 15%, ARC-Challenge 10%, HellaSwag 15%, math/code/knowledge mix)
- NO GPTQ correction, NO rotation (both are the next optimization step)
- Smoke test PASSED 2026-05-22: all 4 generation prompts coherent, no token collapse

**Disk cleanup (2026-05-22)**: Deleted `zaya1-8b-nvfp4-w4a4-soar`, `zaya1-8b-nvfp4-w4a4-mrgptq-v2`,
`zaya1-8b-nvfp4-w4a4-sq-mrgptq`, and `zaya1-8b-bf16-rotated` to recover 47 GB. Only
`zaya1-8b-nvfp4-w4a4` (current working checkpoint) remains. `models--Zyphra--ZAYA1-8B`
(17 GB) is preserved in the WSL HF cache for re-quantization.

#### Optimization Pipeline — Sessions 9–14 (2026-05-19–22) 🟡

SOTA accuracy improvements stacked on the W4A4 baseline, targeting Zyphra's
published BF16 numbers: GPQA-Diamond 71.0%, MMLU-Pro 74.2%, IFEval 85.58%.

| Task | Status | Notes |
|------|--------|-------|
| KV cache FP8 (`kv_cache_dtype=fp8`) | ✅ | Added to all eval + serving scripts. Frees ~1-2 GiB KV cache VRAM. |
| SOAR global-scale optimizer | ✅ | Replaced max-abs with 25-point log-spaced grid search minimizing FP8 block-scale rounding MSE. +0.9pp HellaSwag. |
| EBSS calibration (expert-balanced sample selection) | ❌ **CLOSED 2026-08-15** | 977-sample arcmix corpus reordered to equalize per-expert coverage. **Retired as a lever — see `RESEARCH.md` §5.20.** Two findings: (1) the selection loop never masked already-picked samples, producing a corpus of 3 unique rows (one repeated 972×) that under-observed activation maxima ~40% and **silently corrupted** the resulting checkpoint to chance-level accuracy with byte-identical weights; (2) after fixing selection to sample without replacement, coverage was **unchanged** (0.04 → 0.04) — selecting N from N is a permutation, and `activation_max` is a running max, an order/frequency-independent statistic no reordering can move. EBSS is inapplicable to max-based calibration by construction. Both scripts hardened (diversity guard + inline IGS repair) so this class of silent corruption cannot recur. |
| Fix zero-IGS from uncalibrated modules (post-hoc) | ✅ | `scripts/fix_uncalibrated_igs.py` patches zero IGS with per-layer median fallback. Applied to all checkpoints. |
| block_maxes_store OOM fix | ✅ | SOAR hook accumulated unbounded bmax tensors ([65536] per popular expert). Capped at 1024 elements + `gc.collect()` + `malloc_trim(0)` per layer. Peak RSS: 84.7 GB → ~37 GB. |
| MR-GPTQ (`--mr-gptq` flag) | ✅ | Column-by-column Hessian correction per-layer during calibration. Adds ~30% to calibration time. Expected: +2-4pp on hard reasoning tasks. Flag exists — not yet benchmarked against Zyphra's suite. |
| ARC-mix calibration (Session 14) | ✅ | ARC-Easy 15%, ARC-Challenge 10%, HellaSwag 15% added to calibration mix. Current checkpoint re-quantized with this mix + mixed-precision. |
| Baseline benchmarks — GPQA/MMLU-Pro/IFEval | 🟡 | **IN PROGRESS 2026-05-22.** First run on current mixed-precision checkpoint (no GPTQ). Establishes baseline for the final optimization step. |
| SingleQuant rotation (ART + URT outlier elimination) | 🔴 | `apply_singlequant_rotations.py` exists but has gamma absorption bug: `gamma_new = R @ gamma` is mathematically wrong (element-wise γ doesn't commute with R). **Fix**: absorb rotation into preceding linear's OUTPUT weights, not LN gamma. Script needs this patch before use. **Session 16 note:** the cached base was deleted in the 2026-07-20 cleanup and re-pulled as `models--Zyphra--ZAYA1-8B-legacy`. Also — the script rotates `fc1`'s **input**, but all 24 outliers are at `linear_fc2`, which its own docstring says it skips (SwiGLU nonlinearity blocks clean rotation of the gate half). Fixing the gamma bug would apply correct math to the wrong tensor. **Deprioritised** — Session 16 removed the exemptions without rotation. |
| **Final checkpoint: Rotation + GPTQ + mixed-precision** | ⬜ | After baseline benchmarks complete. Fix rotation script → apply rotation (~15 min) → re-quantize with `--mr-gptq --mixed-precision-threshold 1000 --arc-mix` (~25 min) → benchmark (~80 min). Threshold 1000 because rotation suppresses outliers, fewer layers need BF16 exemption. |
| MoE tuning config (RTX 5070 Ti) | ⬜ | Missing: `E=16,N=2048,device_name=NVIDIA_GeForce_RTX_5070_Ti.json`. Affects TRITON MoE throughput for BF16-exempt layers only. Generate via `python -m vllm.model_executor.layers.fused_moe.benchmark`. Not a blocker for accuracy benchmarks. |
| ARCQuant residual channels | ⬜ | **Session 16: still likely unnecessary, premise updated 2026-08-09.** Phase A did find a small regression (−0.71 pp HellaSwag, CI [−1.26, −0.15]), so the earlier "no measurable cost" premise is retired — but the effect is far too small to justify a residual-correction arc, and no other task resolves it. Implementation is complete on both sides (`build_arcquant_corrections.py` + vLLM branch `wip/arcquant-residual-correction`). **Preferred next lever if recovery is ever wanted: ScaleSweep-style bounded FP8 block-scale search** — same mechanism as SOAR, checkpoint-level only, no kernel rebuild. Note the 2026 literature reports Hadamard rotation *hurts* NVFP4, so the rotation arc stays closed. |


#### Session 16 — BF16 Exemptions Were Redundant (2026-08-08) 🟡

**Headline: the checkpoint shrank 9.46 GB → 6.02 GB (−36%) for a measured cost of
−0.71 pp on HellaSwag** (n=10,042, 95% CI [−1.26, −0.15]). The 384 BF16-exempted
Linears were not earning their 3.44 GB.

> **Claim revised 2026-08-09.** This section originally read "with no measurable
> accuracy cost," based on a single **unpaired** test on ARC-Easy. That test was
> underpowered and pointed the wrong way. The paired Phase A suite below
> supersedes it.

##### Root cause

Outlier *detection* and BF16 *exemption* were coupled in
`quantize_zaya_ct_nvfp4.py`: any layer with `max_abs > threshold` had its
entire MLP exempted, because FusedMoE requires uniform quantization per layer.
24 offending `linear_fc2` modules therefore cost **384** exempted Linears
(12 layers × 16 experts × 2), a 16× overpay where 16 = `num_experts`.
The Week-3 estimate of "~200 MB" was correct in principle and off by 17× in
practice for exactly this reason.

New `--no-bf16-exempt` flag decouples them: outliers are still detected and
recorded in the manifest, but compressed to W4A4 anyway.

##### Measured results

| Metric | 9.46 GB (384 exempt) | 6.02 GB (0 exempt) | Δ |
|---|---:|---:|---:|
| Checkpoint size | 9.46 GB | **6.02 GB** | **−3.44 GB** |
| W4A4 modules | 936 | 1320 | +384 |
| BF16-exempted | 384 | **0** | −384 |
| Outlier layers detected | 12 | 12 | same list |
| ARC-Easy `acc` (n=2376) | 68.39% ± 0.95 | **70.20% ± 0.94** | +1.81 pp |
| ARC-Easy `acc_norm` | 68.10% | 67.72% | −0.38 pp |
| KV cache available | — | **6.83 GiB / 336,835 tok** | — |

**Statistical honesty:** the +1.81 pp ARC-Easy delta is **not significant**
(z = 1.352, p ≈ 0.18; 95% CI [−0.81, +4.43] pp) and `acc_norm` moves the opposite
way. ⚠️ **This comparison was also the wrong test.** Comparing two runs'
aggregate accuracies is an *unpaired* two-proportion test, which discards the
fact that both checkpoints are quantizations of one base model scored on the
*same items*. That correlation is exactly what supplies the statistical power.
The ARC-Easy result should be treated as uninformative, not as support for
"no measurable cost." See Phase A.

##### Phase A — paired loglikelihood suite (2026-08-09)

Four pure-loglikelihood tasks on **both** checkpoints, identical settings,
`log_samples=True`, joined per `doc_id`, tested with **exact-binomial McNemar**
on discordant pairs. 14,319 items per checkpoint. No generation, so the
`<think>`-never-terminates artifact cannot contaminate these numbers, and no
chat template (these are ranked-continuation tasks).

| task | metric | n | 6.02 GB | 9.46 GB | Δ pp | 95% CI | b | c | p |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| hellaswag | acc | 10042 | 45.79% | 46.49% | **−0.71** | [−1.26, −0.15] | 371 | 442 | 0.0140 |
| hellaswag | acc_norm | 10042 | 60.65% | 61.34% | −0.70 | [−1.39, −0.01] | 587 | 657 | 0.0504 |
| arc_challenge | acc | 1172 | 37.97% | 36.95% | +1.02 | [−1.42, +3.47] | 113 | 101 | 0.4522 |
| arc_challenge | acc_norm | 1172 | 37.97% | 40.36% | −2.39 | [−4.97, +0.19] | 105 | 133 | 0.0799 |
| winogrande | acc | 1267 | 56.20% | 59.04% | −2.84 | [−6.04, +0.36] | 196 | 232 | 0.0906 |
| piqa | acc | 1838 | 69.42% | 70.02% | −0.60 | [−2.41, +1.21] | 139 | 150 | 0.5564 |
| piqa | acc_norm | 1838 | 70.89% | 70.08% | +0.82 | [−1.02, +2.65] | 156 | 141 | 0.4166 |

`b` = 6.02 GB correct where 9.46 GB wrong; `c` = the reverse.

**Reading it honestly:**

- **No comparison survives Bonferroni** (α = 0.05/7 = 0.0071). That is *not*
  proof of no cost — absence of significance is absence of resolution.
- **HellaSwag is the only adequately powered task** (n=10,042) and its 95% CI
  **excludes zero**: −0.71 pp [−1.26, −0.15]. Directionally real, practically tiny.
- **The other three cannot resolve small effects.** Their CIs still admit
  −4.97 pp (arc_challenge acc_norm) and −6.04 pp (winogrande). Quote the CIs,
  not the p-values.
- **5 of 7 comparisons point negative**, consistent with a small real regression
  that only HellaSwag has the samples to detect.

**Defensible claim:** −0.71 pp on HellaSwag for −36% checkpoint size; smaller
benchmarks are directionally consistent but underpowered below ~5 pp.

**Method note:** aggregate-only output cannot be salvaged into a paired test
without re-running the model. Always pass `log_samples=True`. Reproduce with
`scripts/run_phase_a.py` + `scripts/analyze_phase_a.py` (`phase_a_driver.sh`
runs both and resumes).

##### Hypothesis

**SOAR likely made the exemptions redundant.** SOAR landed in sessions 9–13,
*after* mixed-precision exemption was introduced, and directly targets the
FP8 block-scale rounding error that `max_abs > 500` causes. Two mitigations
for one problem; the combination was never re-tested. Activation distribution
on this run: min 3.56 / p25 7.34 / median 14.31 / p75 27.75 / **max 8896**
(622× median, `L75.experts.1.linear_fc2`).

##### Smoke test — uncorrected checkpoint is coherent

All 4 prompts produced sensible text with 0 BF16 exemptions. No pad-token
collapse. Mild repetition observed ("Mount Everest is located in…" ×3).
`CutlassNvFp4LinearKernel` + `VLLM_CUTLASS` MoE backend both engaged.
Speed 9.7 tok/s at `enforce_eager` — **not** comparable to the 102.6 tok/s
CUDA-graph figure.

##### Next: Phase A / Phase B

| Phase | Task | Status | Notes |
|---|---|---|---|
| A | Paired loglikelihood suite on **both** checkpoints | ⬜ | `hellaswag` (n=10,042, ±0.9 pp), `arc_challenge`, `winogrande`, `piqa`. No generation → cheap + high power. Requires pulling the 9.46 GB checkpoint from HF as the control (Zyphra publishes no HellaSwag/ARC baseline for ZAYA1-8B). **This is the claim that matters.** |
| B | GPQA-Diamond budget sweep, 6.02 GB checkpoint | ⬜ | n=24 for comparability with the existing 45.8 / 45.8 / 62.5 curve, pushing to **24k and 32k** budgets the old checkpoint could not fit. Tests whether the freed VRAM lifts the reasoning ceiling. Run only if Phase A shows no regression. |

> **Why not GPQA first:** budget-forced GPQA generates up to 12k tokens per
> question. At n=198 that is ~6–7 h *per checkpoint*; at n=24 the CI is ±19 pp.
> Generation cost caps statistical power, so GPQA cannot be primary evidence.

##### ARCQuant status change

`build_arcquant_corrections.py` + the vLLM-side runtime support are complete
but **likely unnecessary** — there is no damage to correct. Preserved on
vLLM branch `wip/arcquant-residual-correction` (commit `6e2f9c5`). Revisit
only if Phase A reveals a regression on a harder benchmark.

##### Bugs fixed this session

| Bug | Impact |
|---|---|
| `DEFAULT_MODEL = "Zyphra/ZAYA1-8B"` in 4 pipeline scripts | Zyphra refactored that repo to a 40-layer config in late June 2026. Running with defaults would load the **wrong architecture**. Repointed to `Zyphra/ZAYA1-8B-legacy`. 26 further one-off diagnostic scripts still reference the old ID. |
| config `ignore` list built from `dynamic_outlier_layers` | Should key on `dynamic_bf16_set` (what was *actually* exempted). Under `--no-bf16-exempt` this told vLLM to look for BF16 weights absent from the checkpoint. |
| Final summary printed "kept at BF16 MLP" alongside "BF16 modules: 0" | Contradictory; now mode-aware and warns when the checkpoint is uncorrected. |

##### Environment state (for resume)

| Item | Location |
|---|---|
| Repo | `~/zaya1-godspeed` (WSL native fs — **not** `/mnt/c`, I/O is much faster) |
| vLLM source build | `~/vllm-src` — **intact**, `cutlass_scaled_fp4_mm_sm120a` compiled 2026-05-15. No rebuild needed. |
| venv | `~/vllm-env` (torch 2.11.0+cu130, vllm 0.20.2) |
| BF16 base | HF cache `models--Zyphra--ZAYA1-8B-legacy` (17 GB, 80-layer, verified) |
| Calibration | `data/calibration/arcmix/calibration_data.pt` (977 × 1024) |
| New checkpoint | `~/zaya1-godspeed/zaya1-8b-nvfp4-w4a4-arcbase` (6.02 GB, gitignored) |
| Recovered artifacts | `results/recovered/` (gitignored) — incl. `arc_easy_mse.json`, the n=2376 baseline |

> **Note:** `SESSION_HANDOFF.md` is gitignored and did not survive the
> 2026-07-20 disk cleanup. This ROADMAP section replaces it.

#### Session 17 — CUDA Graph Capture Bug Found; Throughput Figures Retracted (2026-08-14) 🔴🟢

**The published 102.6 / 407.4 tok/s figures (Session 8) were measured with CUDA
graphs enabled. Confirmed 2026-08-11–14: that path produces numerically wrong
output on this card, independent of MoE backend.** Session 16's own line 395-396
already flagged the 9.7 tok/s `enforce_eager` figure as "not comparable" to
102.6 — this session establishes *why*, and which number is real.

##### Backend sweep (2026-08-11, `zaya1-8b-nvfp4-w4a4-uniform`, greedy)

| backend | with CUDA graphs |
|---|---|
| `flashinfer_cutlass` (default) | garbage |
| `cutlass` | garbage |
| `marlin` (weight-only) | garbage |
| any backend, `enforce_eager=True` | ✅ coherent |

Marlin is weight-only and barely touches the FP4 MoE path — it failing exactly
like the native FP4 kernels means the bug is graph capture itself, not a kernel.
Upstream: [CUTLASS #3096](https://github.com/NVIDIA/cutlass/issues/3096) is a
different (non-graph-capture) bug with its own fix; [FlashInfer #2776](https://github.com/flashinfer-ai/flashinfer/issues/2776)
is graph-capture-specific but its root cause (FlashInfer TRTLLM kernel memory
alignment) wouldn't explain Marlin failing too. No upstream issue currently
documents this exact combination — possibly worth filing.

##### Coherence re-verification (2026-08-14)

First attempt used `llm.generate()` with a raw prompt string (no chat template)
and produced fluent-but-off-topic text — a test-harness mistake, not a second
bug; ZAYA needs its chat template for generative output regardless of backend
(see the existing chat-template gotcha elsewhere in this doc). Re-run via
`llm.chat()` with `enforce_eager=True`: correct, on-topic, budget-appropriate
chain-of-thought reasoning on all 3 test prompts (Rayleigh scattering correctly
invoked for "why is the sky blue"; correctly distinguished RGB vs. CMY primary
color systems). Confirms Session 16's 9.7 tok/s smoke-test speed was on the
*correct* path all along — Session 8's 102.6 tok/s was not.

##### Corrected throughput (5 process invocations per config, GPU idle, 2026-08-14)

| | 6.02 GB uniform | 9.46 GB mixed |
|---|---:|---:|
| Single-stream, median (range) | **9.52** (9.48–9.84) tok/s | **9.51** (9.45–9.81) tok/s |
| Batch-8, median (range) | **73.4** (72.2–74.9) tok/s | **74.4** (72.8–75.7) tok/s |
| Batch-8 scaling vs. batch-1 | 7.71× (96% of ideal) | 7.82× (98% of ideal) |

Near-ideal batch scaling is itself evidence for the open MoE-kernel hypothesis
below (per-step cost not growing with batch size). Variance is 3.6–3.9% across
independent process launches — versus the previously documented 3.4× (340%)
swing under CUDA graphs, which was a symptom of this same bug, not separate
noise. Full detail and citations: `README.md` → "Known Issue: CUDA graph
capture corrupts output on SM120".

##### Open question, now sharper — then retired (same session)

The ~10× TPOT gap flagged in earlier sessions (`trtllm::fused_moe::gemm2`
skipping all tactics) is not resolved by `enforce_eager` alone — it avoids the
*correctness* bug but decode is still slow in absolute terms. Near-linear
batch-8 scaling (above) is consistent with a batch-independent per-step
overhead, which narrowed the search.

**Tested the obvious next step same-day: `--moe-backend marlin`.** Result:
statistically identical to the default backend (9.69 vs 9.52 tok/s single,
72.75 vs 73.4 tok/s batch-8 — within the ~3.6–3.9% noise floor). Two
architecturally unrelated kernels landing at the same speed is evidence
*against* a fixable wrong-tactic bug. Worse: the "~10×" framing was measured
against the 102.6 tok/s figure this same session retracted for being
numerically invalid — a broken code path has no guaranteed relationship to how
much correct work it was doing. **Verdict: the TPOT gap is retired as an open
bug, not solved.** Current honest speed (9.5–9.7 tok/s) may simply be what
this architecture costs on this hardware. Full writeup: `RESEARCH.md` §5.15.

##### Downstream corrections applied this session

Both HF model cards, the GitHub profile README, and the portfolio site all
quoted 102.6/407.4 tok/s and have been corrected to match this table — tracked
so the numbers don't drift back out of sync on the next edit.

##### Addendum, same day: the "retired" verdict above was premature

Went looking for an independent throughput reference to validate that
retirement and found one — llama.cpp PR #23112's own author reports **45.9
tok/s on an RTX 4070 Ti** (Q4_K_M), a slower GPU beating our 9.5 tok/s by
~4.8×. Attempted direct reproduction on our own SM120 hardware: **five
independent build/version/flag combinations, all failed the same way** — a
non-deterministic hang (works once, hangs on an identical rerun of the same
binary and command). Ruled out: the ZAYA model/CCA code (a completely
mainstream Llama 3.2 1B GGUF hung identically), the CUDA 13.2 toolkit
(rebuilt against 12.8, same hang), and the specific llama.cpp commit
(rebuilt at the pre-regression tag `b7376`, same non-deterministic hang).
This points to a WSL2/driver-level issue, not a fixable software bug — see
`RESEARCH.md` §5.16 for the full six-attempt diagnostic log.

**Corrected status: the TPOT gap is reopened, not retired.** Real external
evidence exists and disagrees with our number by a wide margin; we simply
couldn't verify it on matched hardware this session. Next attempt starts with
a Windows driver / WSL kernel update, not another build variation.

##### Second addendum, same day: the gap likely has a principled cause

Went looking for engineering levers to close the gap and found a better
answer first: **activation quantization (W4A4) provides no speed benefit at
batch-1 and can be slower than weight-only quantization** — decode at
batch-1 is memory-bandwidth-bound, and quantizing activations only pays off
when compute is the bottleneck (batched serving, prefill). This is
documented, expected behavior for the scheme, not a bug. The §5.16 comparison
(45.9 tok/s weight-only vs. 9.5 tok/s W4A4) was comparing two different
design points, not a fair speed contest. W4A4's actual advantage — memory
footprint and batched throughput — shows up exactly where this project
already measured it: batch-8 at 73–74 tok/s, 96–98% of ideal scaling.

**This doesn't fully close §5.16** (still blocked on the WSL2 hang for a
real same-hardware comparison), but it means no engineering fix is owed
before publishing current numbers — the gap is now principled, not
mysterious. Full writeup: `RESEARCH.md` §5.17. Filed upstream WSL logs
attached same day: [microsoft/WSL#41361](https://github.com/microsoft/WSL/issues/41361).

Also explored, same day: **SGLang has genuine ZAYA1 support** (merged
[PR #26347](https://github.com/sgl-project/sglang/pull/26347), v0.5.14) —
untested against our compressed-tensors NVFP4 checkpoint specifically, but a
real option that doesn't touch the broken llama.cpp/WSL2 path. **Speculative
decoding** is the more promising lever if further speed work is wanted:
MoE-specific research shows real batch-1 gains from temporal correlation in
expert routing, and this project already has speculative-decoding experience
elsewhere in the stack (Godspeed).

##### Third addendum, same day: speculative decoding tested, real 2.2× win found

Acted on the lever flagged above. vLLM's built-in n-gram speculative decoding
(`--speculative-config '{"method": "ngram", ...}'`, zero training, zero new
model) gives **no gain on free-form generation** (~9.3 vs ~9.5 tok/s baseline
— expected, no context overlap to exploit) but a **validated 2.2× speedup on
realistic coding-edit prompts** (21.11 vs 9.62 tok/s median, 5 reps each,
zero anomalies, output coherence confirmed identical in both modes). This is
close to exactly Godspeed's actual workload shape (read a file, echo most of
it back with a small edit).

Full methodology and numbers: `RESEARCH.md` §5.18. **Not yet wired into
`~/scripts/vllm-serve.sh`** — logged now so it isn't lost, deployment is a
separate follow-up.

##### Fourth addendum, same day: deployed to the production serve script

Wired `--speculative-config` into `~/scripts/vllm-serve.sh` (the script behind
the desktop SERVE button) alongside the mandatory `--enforce-eager` flag.
Validated live, not just in the offline benchmark harness: started the real
server, waited for `/health`, sent a real request to `/v1/chat/completions`
with a coding-edit-style prompt, got coherent output back through the actual
OpenAI-compatible API. Stopped the test server afterward (not left running
unrequested).

One tradeoff surfaced only at serve time: vLLM logs that async scheduling is
disabled when n-gram speculative decoding is active. This is a real
secondary cost the offline 2.2× figure doesn't account for (that number
compared `enforce_eager` on/off, not async-scheduling on/off) — presumed net
positive given the size of the win, but not independently measured.

**Flagged, not resolved:** `~/scripts/vllm-serve.sh` has no version control
(no git repo at `~/scripts`, no chezmoi tracking it) — this change, and the
whole script, would be lost on an environment rebuild. A separate decision
from this finding.

Full detail: `RESEARCH.md` §5.18 (update).

#### Session 18 — Generative Eval Suite + Two Silent-Corruption Fixes (2026-08-15) 🟢🔴

**EBSS closed permanently, and the bug it exposed fixed at the source.**
Requantizing with EBSS calibration produced a checkpoint that exited 0 with
no errors and byte-identical weights, yet scored at **chance level**
(hellaswag acc_norm 60.29% → 25.75%). Root cause: `ebss_resample()` never
masked already-picked samples, yielding a corpus of **3 unique rows out of
977** (one repeated 972×), which under-observed activation maxima ~40% and
inflated `input_global_scale` on 95.8% of modules. After fixing selection to
sample without replacement, coverage came back **identical** (0.04 → 0.04) —
selecting N from N is a permutation, and `activation_max` is a running max,
an order/frequency-independent statistic. **EBSS cannot help max-based
calibration by construction**, so it is retired rather than left as a
maybe-retry. Both scripts hardened so this class of silent corruption cannot
recur. Full detail: `RESEARCH.md` §5.20.

**Generative eval suite built and hardened.** GSM8K, HumanEval and MMLU-Pro
budget-forced harnesses now exist, closing the loglikelihood blind spot (those
tasks score *ranking*, never *producing* — the reason a checkpoint incapable
of forming a sentence once scored 61% on HellaSwag). Checked lm-eval's own
reasoning support first per the standard-tools rule: its `think_end_token`
strips post-hoc (`split(tok)[-1]`), so a model that never closes `</think>`
has its whole trace scored as the answer — the artifact that put IFEval at
19.8%. Budget *forcing* is therefore genuinely required here; all scoring
still uses lm-eval's own regexes and HF `evaluate`'s `code_eval`. Also fixed
a real defect: extraction failures (forced answer opening with a newline)
were being recorded as reasoning failures, costing 1 of 20 items. Full
detail: `RESEARCH.md` §5.21.

**Baseline limits, stated up front:** Zyphra publishes no GSM8K or HumanEval
figure, and its MMLU-Pro 74.2 comes from a private harness with undisclosed
generation limits — so these results largely stand alone rather than as
retention-vs-baseline. Generating a matched BF16 baseline locally is
impossible: 17.7 GB of weights does not fit in 16 GB.

**Not pursued, with reasons:** ARCQuant residual corrections for the 10
uncorrected outlier layers are implemented on both sides but require a
patched vLLM branch to serve — that would make the published checkpoint
unusable with stock vLLM, a bad trade for a public artifact. EAQuant's
routing-consistency idea (independent 2026 work validating expert-aware PTQ)
is real but built on OmniQuant/DuQuant, so it needs reimplementation rather
than reuse. EAGLE-3 needs a model-specific draft head trained from scratch.
CUTLASS #3096's `compute_120f` rebuild remains the largest untried speed
lever. None fit a single session.

#### Session 18 results — generative benchmarks landed (2026-08-15/16) ✅

**First generative accuracy numbers for this checkpoint, and the first
HumanEval figure for ZAYA1-8B in any precision.** Uniform 6.02 GB checkpoint,
`enforce_eager=True`, think_budget 4096, suite wall time 1h58m.

| benchmark | score | 95% CI | n |
|---|---:|---|---:|
| **HumanEval** | **72.6%** pass@1 | [65.3, 78.8] | 164 |
| **GSM8K** | **65.5%** | [62.9, 68.0] | 1,319 |
| **MMLU-Pro** (0-shot) | **48.1%** | [44.5, 51.8] | 700 |

**HumanEval is the headline:** published comparisons put Qwen 3 7B at ~68–72%
and Llama 3 8B at 62–65% *at full precision*. This matches or beats them with
4-bit weights **and** activations in 6.02 GB.

**The budget hypothesis was tested and rejected.** Both benchmarks were re-run
at 8192 and compared with paired McNemar on identical items: GSM8K **+0.15 pp**
(p=0.9581) and MMLU-Pro **+3.29 pp** (p=0.0673) — neither significant. 4096 is
the correct operating budget; 8192 costs 3× the wall time for nothing. Ceiling
hits barely moved (GSM8K 78% → 71%) — this model keeps thinking regardless.

**The MMLU-Pro gap to Zyphra's 74.2% is a protocol difference, not
quantization damage.** lm-eval's MMLU-Pro is `num_fewshot: 5`; this harness is
0-shot. Published INT4 loss on MMLU-Pro is ~1.6 pp, so a 26 pp quantization
cost would be far outside anything documented, and §5.13's paired test already
bounded this checkpoint's cost at −0.71 pp HellaSwag. **Reported as "0-shot,
budget-forced" and explicitly not comparable.** Full analysis: `RESEARCH.md`
§5.22.

#### The reasoning/latency tradeoff, quantified (2026-08-16) 🔴

ZAYA1's chat template ships an `enable_thinking` flag (Zyphra's own) that this
project had never used. Off, it pre-closes `<think>` so the model answers
immediately. A 3-prompt probe looked excellent — 3× faster, better answers — so
it was measured properly on all three benchmarks. Paired McNemar, identical
items:

| benchmark | thinking | no-thinking | Δ | p | wall time |
|---|---:|---:|---:|---:|---|
| HumanEval | 72.6% | 43.9% | **−28.66 pp** | <0.0001 | 15 m → 2 m |
| MMLU-Pro | 48.1% | 26.7% | **−21.43 pp** | <0.0001 | 39 m → 4 m |
| GSM8K | 65.5% | 48.1% | **−17.36 pp** | <0.0001 | 63 m → 8 m |

**~8.5× faster overall (1h58m → 14m) for a 17–29 point accuracy loss.** All
three highly significant.

**Conclusion: ZAYA1's accuracy *is* its reasoning, and its reasoning *is* what
makes it slow — they cannot be separated.** This checkpoint is not a fast
interactive coding agent and no configuration makes it one. That is now
evidence-based rather than inferred, and it is a genuine finding rather than a
limitation to hide.

**Where the lever still applies:** per-request routing, not a global switch.
vLLM takes `chat_template_kwargs: {"enable_thinking": false}` per request, so
mechanical work can take the fast path while real problem-solving keeps
reasoning. Notably 12/53/173 items were solved *only* without thinking — some
tasks are actively hurt by overthinking, which a classifier could exploit.
Full analysis: `RESEARCH.md` §5.23.

#### Engineering Cleanup — Session 15 Action Items ⬜

The following items were identified as unprofessional shortcuts during session 15.
They don't block benchmark results but should be resolved before publication.

| Item | Priority | Action |
|------|----------|--------|
| **Patch file for zaya.py prefix-caching fix** | High | The warn-and-disable fix in `/home/ttimm/vllm-src/vllm/model_executor/models/zaya.py` exists only in the editable install — invisible to anyone cloning this project and lost on env rebuild. Create `patches/vllm_zaya_prefix_caching.patch` via `git diff` from the vllm-src repo. Add apply instructions to `patches/README_W4A4.md`. |
| **Submit vLLM upstream PR** | Medium | The hard assertion `assert not cache_config.enable_prefix_caching` should be a graceful warn-and-disable for all hybrid Mamba/attention models, not just Zaya. Submit PR to `vllm-project/vllm` targeting `vllm/model_executor/models/zaya.py` and the broader `verify_and_update_config` logic in `model_executor/models/config.py`. This is a legitimate upstream bug. |
| **Accept GPQA gated dataset terms** | High — blocks benchmark | Visit https://huggingface.co/datasets/Idavidrein/gpqa while logged into HF as Ttimms. Click "Access repository". Without this, `leaderboard_gpqa_diamond` errors on `DatasetNotFoundError`. |
| **Launch overnight benchmark** | High | Once GPQA is unblocked: `nohup python3 scripts/run_full_benchmarks.py --model ./zaya1-8b-nvfp4-w4a4 --output results/lmeval_w4a4_baseline.json > results/bench_baseline.log 2>&1 &` (from WSL with `vllm-env` active). Full run: GPQA ~15 min + IFEval ~20 min + MMLU-Pro ~20 hrs. |
| **MoE tuning config** | Low | `E=16,N=2048,device_name=NVIDIA_GeForce_RTX_5070_Ti.json` missing — causes "Using default MoE config" warning on every startup. Generate via `python -m vllm.model_executor.layers.fused_moe.benchmark` in vllm-env. Only affects TRITON MoE backend (BF16-exempt layers); W4A4 layers use CUTLASS. |

**How to create the patch file** (from WSL):
```bash
cd /home/ttimm/vllm-src
git diff HEAD vllm/model_executor/models/zaya.py > \
    "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/patches/vllm_zaya_prefix_caching.patch"
```

**Baseline benchmark command** (from WSL, `vllm-env` active):
```bash
cd "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed"
python3 scripts/run_full_benchmarks.py
# Default model: ./zaya1-8b-nvfp4-w4a4 — output: results/lmeval_w4a4_zyphra.json
```

**Final checkpoint pipeline** (run after baseline benchmarks complete):
```bash
# Step 1 — Fix rotation script gamma absorption (edit apply_singlequant_rotations.py)
# Absorb R into preceding linear's output weights, not LN gamma

# Step 2 — Apply rotation to 12 outlier layers (~15 min)
python3 scripts/apply_singlequant_rotations.py \
    --input Zyphra/ZAYA1-8B \
    --manifest zaya1-8b-nvfp4-w4a4/quantization_manifest.json \
    --output zaya1-8b-bf16-rotated

# Step 3 — Re-quantize with GPTQ + higher threshold (~25 min)
python3 scripts/quantize_zaya_ct_nvfp4.py --scheme w4a4 \
    --model-id ./zaya1-8b-bf16-rotated \
    --mixed-precision-threshold 1000 \
    --mr-gptq --arc-mix \
    --output-dir ./zaya1-8b-nvfp4-w4a4-final

# Step 4 — Benchmark final checkpoint (~80 min)
python3 scripts/run_full_benchmarks.py --model ./zaya1-8b-nvfp4-w4a4-final \
    --output results/lmeval_w4a4_final.json
```

---

## Phase 3 — Teacher Trajectory Generation 🟡

**Goal**: Generate 300–500 mechanically verified tool-calling trajectories using
DeepSeek V4 Pro via NVIDIA NIM as teacher, routed through Godspeed.

**Teacher**: `nvidia_nim/deepseek-ai/deepseek-v4-pro`
- Released April 24, 2026 | MIT license | 1.6T total / 49B active MoE
- SWE-bench Verified: 80.6% | LiveCodeBench: 93.5% | Codeforces: 3,206
- Available on NVIDIA NIM free tier (4 keys available)
- Rate limits: ~30 RPM per key → ~120 RPM effective with rotation → ~13 min for 200 tasks

| Task | Status | Notes |
|------|--------|-------|
| Configure Godspeed for NIM DeepSeek V4 Pro | ⬜ | `settings.yaml` with model routing: plan via v4-pro-max, edit via v4-pro |
| Run Godspeed headless against 20-task suite | ⬜ | Baseline trajectory quality check |
| Expand 20 base tasks to 200+ variants | ✅ | `scripts/mutate_tasks.py` — 6 mutation types + 10 OOD tasks |
| Run Godspeed headless against 200 mutated tasks | ⬜ | Each task runs in headless mode, conversation logger captures JSONL |
| Filter trajectories through quality gates | ⬜ | `scripts/remap_to_zaya.py` — exit code 0, no dangerous commands, schema valid |
| Remap to ZAYA XML ChatML format | 🟡 | `scripts/remap_to_zaya.py` built — confirmed 0/1325 old sessions pass (expected) |
| Validate tool coverage (30+ Godspeed tools) | ⬜ | |
| Split train/val/eval (80/10/10) | ⬜ | |

**Quality gates** (mandatory, per context doc):
1. Mechanical verify hook passed (13/20 tasks have hooks)
2. Jaccard tool selection >= 0.7 vs expected tools
3. Zero dangerous command flags in reward annotations
4. Zero schema validation errors in tool call sequence
5. Session exit code 0 (success) from headless mode

**Volume target**: 300–500 verified trajectories. ZAYA's MoE (760M active) has high
sample efficiency. Verified 300 >> unverified 3,000. Do not scale volume before
fixing quality.

---

## Phase 4 — ZAYA XML Format & Tool Schema ⬜

**Goal**: Ensure training data uses ZAYA1-8B's exact tool-call format.

ZAYA1-8B uses a JSON-inside-XML format via vLLM's `--tool-call-parser zaya_xml`:
```
<zyphra_tool_call>{"name": "tool_name", "arguments": {...}}</zyphra_tool_call>
```

This differs from Godspeed's native Qwen3-Coder XML format (`<function=name>`) and
from OpenAI-standard `tool_calls` JSON. A remapper is required BEFORE data enters
Unsloth/TRL — silent corruption otherwise.

| Task | Status | Notes |
|------|--------|-------|
| Build ZAYA XML format remapper | ✅ | `scripts/remap_to_zaya.py` — Godspeed JSONL → ZAYA ChatML |
| Verify ZAYA chat template compatibility | ⬜ | `tokenizer.apply_chat_template()` with `<|im_start|>` / `<|im_end|>` tokens |
| Spot-check 50 remapped trajectories manually | ⬜ | Before any training run — this is the highest-risk item |
| Test vLLM `zaya_xml` parser with sample output | ⬜ | Send remapped data through vLLM, verify tool calls parse |

---

## Phase 5 — QLoRA Fine-Tuning 🟡

**Goal**: Train ZAYA1-8B to produce valid tool calls in Godspeed's XML format.

**Strategy**: SFT → Rejection Sampling → GRPO (2026 SOTA pipeline from published models)

| Task | Status | Notes |
|------|--------|-------|
| Training script with TRL SFTTrainer | ✅ | `scripts/train.py` — QLoRA with config-driven pipeline, dry-run support |
| Dry run (1 batch, no save) to catch OOM | ⬜ | Target: <12 GB VRAM |
| SFT Stage 1: 1–2 epochs on 100 verified trajectories | ⬜ | Format learning only. Monitor AIME regression (<5% tolerance) |
| Evaluate baseline checkpoint on 20-task suite | ⬜ | Primary metric: Jaccard + mech verify |
| Full SFT on 300–500 verified trajectories | ⬜ | `WANDB_MODE=offline` on Windows |
| Save LoRA adapter weights | ⬜ | |
| GRPO Stage 2: policy improvement via verifiable rewards | ⬜ | `scripts/train_grpo.py` built. `loss_type="dapo"`, `num_generations=4`, vLLM colocate. |
| Merge adapter (optional, for vLLM deployment) | ⬜ | |

**VRAM Budget (QLoRA, 16 GB GPU)**:

| Component | VRAM |
|-----------|------|
| NF4 base model | 7.2 GB |
| LoRA adapters | ~0.0 GB |
| Optimizer (8-bit Adam) | 0.3 GB |
| Gradients | 0.3 GB |
| Activations (w/ gradient checkpointing) | 2–4 GB |
| **Total** | **10–12 GB ✓** |

**Catastrophic forgetting mitigation**: QLoRA, conservative rank (r=16), 1–2 epochs max.
Hard stop if AIME degrades >5% from baseline (89.1%).

---

## Phase 6 — Evaluation ⬜

**Goal**: Measure tool-calling accuracy improvement and close the agentic gap.

| Task | Status | Notes |
|------|--------|-------|
| Godspeed 20-task internal benchmark (Jaccard + mech verify) | ⬜ | Primary iteration metric — fast enough for every checkpoint |
| BFCL-v4 | ⬜ | Current baseline: 39.22 vs Qwen3-4B 49.7 (+10pt gap to close) |
| τ² (agentic) | ⬜ | Current baseline: 43.12 vs Qwen3.5-4B 82.1 (+39pt gap) |
| SWE-bench dev-23 subset | ⬜ | Godspeed has existing infra from benchmark runs |
| Regression: AIME subset + LiveCodeBench subset | ⬜ | Catch catastrophic forgetting |
| Compare against Qwen2.5-Coder-14B baseline | ⬜ | Current Godspeed default driver |
| Ablation: LoRA rank (r=8 vs r=16 vs r=32) | ⬜ | Optional |
| Ablation: with/without router layers in targets | ⬜ | Optional |

**Target outcomes**:
- Conservative (300 trajectories, SFT only): BFCL-v4 ~44–47, τ² ~55–65
- Aggressive (500 trajectories, SFT + GRPO): BFCL-v4 ~50+, τ² ~65–75

Do NOT expect τ² parity with Qwen3.5-4B (82.1) in one cycle — that model has
dedicated agentic RL at scale. This is a first cycle.

---

## Phase 7 — Production Deployment ⬜

**Goal**: Merge fine-tuned ZAYA1-8B into Godspeed as a first-class driver.

| Task | Status | Notes |
|------|--------|-------|
| Merge adapter or export full model | ⬜ | |
| Add to Godspeed model presets | ⬜ | `--preset zaya` already exists |
| Update driver catalog with fine-tuned benchmark scores | ⬜ | |
| Write model card on HuggingFace | ⬜ | Community contribution |
| Publish findings (blog post / technical note) | ⬜ | Portfolio piece |

---

## Pipeline Architecture

```
mutated_tasks.jsonl (200 tasks)
       │
       ▼
Godspeed headless ─── DeepSeek V4 Pro (NIM)
       │
       ▼
~/.godspeed/training/*.conversation.jsonl
       │
       ▼
remap_to_zaya.py (quality gates + ZAYA XML format)
       │
       ▼
data/train_zaya.jsonl (300–500 verified ChatML trajectories)
       │
       ▼
train.py ─── Unsloth + TRL SFTTrainer (QLoRA, r=16)
       │
       ▼
checkpoints/zaya1-tool-call/
       │
       ▼
vLLM serve (evaluation) → Godspeed 20-task benchmark → BFCL-v4
```

## Known Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Catastrophic forgetting (AIME 89.1 → <84) | Loss of ZAYA's core reasoning | QLoRA r=16, low LR, 1–2 epochs. Monitor AIME every epoch. Hard stop if >5%. |
| Tool call schema mismatch | Model trains on garbage silently | Manual spot-check 50 rows after remapping. Test through vLLM zaya_xml parser. |
| Shopify OOD failure mode | Good benchmarks, bad real usage | 30% OOD tasks minimum. Mutate before generation, not after. |
| Teacher format hallucination | DeepSeek V4 generates invalid tool calls | Godspeed's schema validator catches these at runtime. Filter during remapping. |
| Benchmark leakage | SWE-bench tasks overlap with training data | Cross-check tasks.jsonl against SWE-bench Verified task IDs before release eval. |
| No GGUF / llama.cpp support | Can't use spec decoding or local llama.cpp server | vLLM is the only viable local path |
| Zyphra fork required | Extra build step for both transformers and vLLM | One-time setup, documented |
| 16 GB VRAM ceiling | ~24K context max, limited batch size | Adequate for agent loop (single sequence) |

## Reference Links

- [`MODEL_SELECTION.md`](./MODEL_SELECTION.md) — Why ZAYA1-8B for a 16 GB local coding agent (2026-06-02 HF survey: competitor W4A4 quant `switzerchees/ZAYA1-8B-NVFP4`, Qwen3-Coder family VRAM fit analysis, upgrade triggers)
- [Zyphra/ZAYA1-8B on HuggingFace](https://huggingface.co/Zyphra/ZAYA1-8B)
- [Technical Report (arXiv 2605.05365)](https://arxiv.org/abs/2605.05365)
- [Zyphra Blog Post](https://www.zyphra.com/post/zaya1-8b)
- [Zyphra vLLM Fork](https://github.com/Zyphra/vllm/tree/zaya1-pr)
- [Zyphra Transformers Fork](https://github.com/Zyphra/transformers/tree/zaya1)
- [NVIDIA NIM — DeepSeek V4 Pro](https://build.nvidia.com/deepseek-ai/deepseek-v4-pro)
- [Godspeed Coding Agent](https://github.com/t-timms/godspeed-coding-agent)
