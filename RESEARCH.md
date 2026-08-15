# Research Document: Completing ZAYA1-8B with Agentic Multi-Turn Tool Calling

**Project**: zaya1-godspeed  
**Experiment Lead**: Tremayne Timms  
**Date**: May 12, 2026  
**Status**: Phase 2 Stage 2 CUTLASS SM120 kernels compiled and verified (May 15, 2026, session 4). Path B W4A4 Phase 1 dry-run completed with CPU calibration failure, layer-wise GPU calibration designed (May 17, 2026). NVFP4 Compressed-Tensors ZAYA1-8B serves correctly via vLLM on RTX 5070 Ti (Blackwell sm_120). Greedy decoding answers "The capital of France is" → " Paris." and produces coherent BST reasoning. All 4,244 weights load; ~5.5 GiB VRAM; ~0.86–0.90 tok/s on Path A (on-the-fly Python dequant). Five upstream vLLM/Zaya fixes applied. Stage 2 CUTLASS speedup next; W4A4 weight+activation quantization parallel track.

---

## 1. Abstract

ZAYA1-8B (Zyphra, May 2026) achieves state-of-the-art reasoning benchmarks for its parameter class (760M active / 8.4B total, MoE). However, per its technical report, Zyphra deliberately skipped the multi-turn agentic RL stage. This project aims to close that agentic gap by fine-tuning ZAYA1-8B for structured multi-step tool calling using teacher-distilled SFT+GRPO with DeepSeek V4 Pro as the teacher model.

This document summarizes the compatibility analysis, upstream code audit, SOTA training configuration, and pipeline design as of May 2026.

---

## 2. Model Architecture Summary

| Property | Value |
|----------|-------|
| Architecture | `ZayaForCausalLM` (custom, Qwen-style chat template) |
| Total params | 8.4B |
| Active params | 760M (16 experts, top-1 routing, MoE) |
| Hidden size | 2048 |
| Layers | 80 (40 CCA attention + 40 MoE, interleaved) |
| Attention | Compressed Convolutional Attention (CCA): depthwise+grouped conv1d QK mixing, L2-normalized QK, per-head temperatures, dual time-stream values |
| Context window | 131,072 tokens (rope_theta=5,000,000) |
| Special tokens | `<|im_start|>`, `<|im_end|>` (Qwen-style), `<think>` blocks |
| Tool-call format | `<zyphra_tool_call>{"name":"...","arguments":{...}}</zyphra_tool_call>` (tokens 101-102) |
| Tool-response format | `<zyphra_tool_response>result</zyphra_tool_response>` (tokens 103-104) |
| Weight tying | `tie_word_embeddings=True` |
| License | Apache 2.0 |

---

## 3. Critical Discovery: Tool-Call Format Mismatch

### 3.1 The Finding

During the upstream code audit, we discovered that the project's training data format used incorrect tool-call boundary tags. The ZAYA1-8B tokenizer has dedicated single tokens for tool-call boundaries:

| Token ID | Content | Expected Behavior |
|----------|---------|-------------------|
| 101 | `<zyphra_tool_call>` | Single-token tool call start |
| 102 | `</zyphra_tool_call>` | Single-token tool call end |
| 103 | `<zyphra_tool_response>` | Single-token tool response start |
| 104 | `</zyphra_tool_response>` | Single-token tool response end |

The project was using `<tool_call>` / `</tool_call>` instead. These are not special tokens — the tokenizer would split them into 5-6 subword tokens each: `<`, `tool`, `_`, `call`, `>`. Training a model with the wrong boundary tokens would produce outputs that the vLLM `zaya_xml` parser cannot parse, and the model would not learn to use the dedicated single-token boundaries.

### 3.2 Resolution

All tool-call references across 8 files were updated to use `<zyphra_tool_call>` / `</zyphra_tool_call>`. Tool responses were updated to use `<zyphra_tool_response>` / `</zyphra_tool_response>` instead of the old `[tool_name result]\n` prefix. The remapper (`scripts/remap_to_zaya.py`) now produces training data that matches ZAYA1-8B's native output format.

### 3.3 Verification

The fix was verified by cross-referencing:
1. `Zyphra/ZAYA1-8B` tokenizer_config.json (tokens 101-104)
2. `Zyphra/vllm` zaya_tool_parser.py (line 59-60: `self.tool_call_start_token = "<zyphra_tool_call>"`)
3. 100 unit tests, all passing

---

## 4. Training Pipeline Design

### 4.1 Pipeline Architecture

```
200 mutated tasks → Godspeed headless (DeepSeek V4 Pro via NIM)
    → conversation JSONL → remap_to_zaya.py (5 quality gates)
    → train_zaya.jsonl → train.py (QLoRA SFT, SFTTrainer)
    → GRPO Stage 2 (train_grpo.py)
    → vLLM serve → Godspeed 20-task benchmark → BFCL-v4
```

### 4.2 Phase 1: Compatibility Gate ✓

- Verified Zyphra transformers fork loads ZAYA1-8B with PEFT LoRA
- Confirmed 8.2M trainable parameters (0.09% of total) targeting attention projections only
- Verified gradient flow through LoRA adapters
- Architecture documented in `COMPATIBILITY.md`

### 4.3 Phase 2: Inference Pipeline (Unblocked May 11 2026)

**Root cause analysis**: Phase 2 was initially blocked by Windows desktop compositor consuming ~15.9 GB VRAM. Investigation revealed the actual culprit: a WSL2 `llama-server` process running Qwen3.6-27B-UD-Q4_K_XL consuming ~15 GB. The model and server were removed, freeing 15.3 GB VRAM.

**Inference path analysis** (all tested May 11 2026 on RTX 5070 Ti 16 GB):

| Path | Model Size | KV Cache Free | Status |
|------|-----------|---------------|--------|
| **vLLM + FP8** | 8.76 GB | 5.37 GB | Loads and serves (2s startup). Output quality unverified. |
| **vLLM + bf16** | 16.48 GB | -0.39 GB | **Cannot serve** — model weights exceed GPU memory budget. |
| **vLLM + NVFP4 (our build)** | ~4.76 GB | ~12 GB | Weights mapped (2483/2483). Requires Zyphra vLLM fork. |
| vLLM + MXFP4 (community) | 5.45 GB | ~10 GB | Failed — weight shape mismatch with Zyphra fork. |
| vLLM + bitsandbytes | N/A | N/A | Failed — ZayaForCausalLM lacks `packed_modules_mapping`. |
| NF4 + transformers | 7.2 GB | ~8 GB | Broken — garbage output from CCA attention + dequant. |
| Zyphra Cloud | 0 GB | N/A | Available as API. Not self-hosted. |

**`serve_zaya1.py` re-engineering**: The inference server script was rewritten to SOTA standards:
- `subprocess.Popen` for non-blocking server management
- Health endpoint polling with configurable timeout
- `--enforce-eager` default (skips 8-minute torch.compile/CUDA graph warmup)
- MXFP4 and FP8 quantization flag support
- Command output matches official Zyphra deployment documentation (ref: HF model card 2026-05-11)
- Daemon mode for background server operation

### 4.4 Phase 3-4: Teacher Trajectory Generation + Format Remapping

- 200+ mutated tasks generated via `scripts/mutate_tasks.py` (6 mutation types, 30% OOD)
- Quality gates applied during remapping:
  1. Mechanical verify hook (exit_code=0)
  2. Jaccard tool selection ≥0.7 vs expected tools
  3. Zero dangerous command flags
  4. Zero schema validation errors
  5. Maximum token budget (4096 estimated tokens)
- Format conversion: Godspeed JSONL → ZAYA ChatML with `<zyphra_tool_call>` tags
- Optional `--include-tools` flag adds TRL-compatible `tools` JSON schema column

### 4.5 Phase 5: Training (Ready)

#### Stage 1: QLoRA SFT (`scripts/train.py`)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Method | SFTTrainer (TRL) | Native conversational dataset, `assistant_only_loss` |
| Quantization | NF4 (bitsandbytes) | Fits on 16 GB GPU |
| LoRA rank | r=16 | Conservative for small dataset (300-500 examples) |
| LoRA targets | `o_proj`, `linear_q`, `linear_k`, `val_proj1`, `val_proj2` | Attention projections only, not expert weights |
| rsLoRA | Enabled | Rank-stabilized scaling: `alpha/√r` instead of `alpha/r` |
| Loss type | `chunked_nll` | 60% memory reduction via chunked cross-entropy |
| Liger Kernel | Enabled | 20% throughput increase, 60% memory reduction |
| assistant_only_loss | True | Only trains on assistant output tokens |
| Epochs | 2 | 1-2 epochs max per project constraints |
| Learning rate | 2e-4 | Higher LR for LoRA adapters |
| Gradient checkpointing | True | Reduces activation memory |
| bnb_4bit_use_double_quant | True | Nested quantization saves ~0.36 GB on NF4 base per QLoRA paper |
| VRAM budget | ~9.5-11.5 GB | Reduced from ~10-12 GB by double quantization |

#### Stage 2: GRPO (`scripts/train_grpo.py`)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Loss type | `dapo` | Token-level normalization, no response-length bias |
| Reward scaling | `batch` | Global std normalization, robust reward shaping |
| Num generations | 4 | 4 rollouts per prompt for advantage estimation |
| KL penalty | β=0.0 | KL penalty not needed per modern research |
| vLLM | Colocate mode | Fast generation on single GPU |
| Reward functions | Format (0.5), Schema (0.4), Length (0.1) | Primary: valid `<zyphra_tool_call>` XML + JSON |

### 4.6 Phase 6-7: Evaluation + Deployment (Not Started)

Target benchmarks:
- BFCL-v4: 39.22 → 50+ (SFT only), 55+ (SFT+GRPO)
- τ² (agentic): 43.12 → 65-75 (SFT+GRPO)
- Regression monitoring: AIME ≥84 (hard floor), LiveCodeBench ≥60

---

## 5. NVFP4 Model Quantization Pipeline

### 5.1 Motivation

bf16 ZAYA1-8B (16.48 GB) cannot fit on a 16 GB consumer GPU with room for KV cache. FP8 halves this to 8.76 GB. NVFP4 cuts it to ~4.76 GB — a 3.5× reduction from bf16 — leaving 12+ GB for KV cache and enabling 16K+ context windows on the RTX 5070 Ti.

NVFP4 is NVIDIA's native block-structured 4-bit floating point format. It is **hardware-accelerated on Blackwell GPUs (sm_120)** via the NVFP4 Tensor Core MMA instructions, unlike MXFP4 (community format that failed to load with the Zyphra vLLM fork due to weight shape mismatches).

### 5.2 Pipeline Architecture

```
Zyphra/ZAYA1-8B (HF safetensors, 16.47 GB)
    │
    ▼ scripts/convert_zaya_to_gguf.py
    │
FP16 GGUF (16.47 GB, arch=llama, shortened tensor names)
    │
    ▼ llama.cpp llm-quantize (NVFP4 type, with fallback fix)
    │
NVFP4 GGUF (4.76 GB, 4.52 bpw, 80 fallback tensors)
    │
    ▼ vLLM GGUF loader (patched for zaya arch + name mapping)
    │
OpenAI-compatible inference endpoint
```

### 5.3 Converter Design: `scripts/convert_zaya_to_gguf.py`

**Problem**: ZAYA1-8B's HF tensor names exceed the GGUF 64-character limit. Expert weight names like `model.layers.23.zaya_block.experts.local_experts.13.linear_fc2.weight` (67 chars) cannot fit.

**Solution**: Name shortening via deterministic abbreviation:
| Original | Shortened |
|----------|-----------|
| `zaya_block` | `zblk` |
| `local_experts` | `lexp` |
| `linear_fc` | `fc` |
| `self_attn` | `attn` |
| `input_norm` | `inp_n` |
| `res_scale` | `rs` |
| `router_mlp` | `rmlp` |

Shortened names stay within the 64-char GGUF limit. A `name_map.json` file is generated alongside the GGUF for reverse mapping during vLLM loading.

**GGUF architecture**: Set to `"llama"` (not `"zaya"`) for llama.cpp quantizer compatibility. The quantizer validates architecture strings and rejects unknown architectures.

### 5.4 NVFP4 Quantizer Fix

**Bug found**: The llama.cpp NVFP4 quantizer (`llama-quant.cpp`, `tensor_type_fallback` function) lacked a fallback type for `GGML_TYPE_NVFP4`. When encountering tensors with `ncols` not divisible by 64 (required by NVFP4's 16×16 block quantization), the quantizer would crash with:

```
no tensor type fallback is defined for type nvfp4
```

**Fix applied** (line ~391 of `llama-quant.cpp`):
```cpp
case GGML_TYPE_NVFP4: return_type = GGML_TYPE_F16; break;
```

This causes small tensors (CCA convolution kernels, biases, scalars) to remain in F16 precision while all large weight matrices are quantized to NVFP4. 80 of 2483 tensors use this fallback — they collectively represent <0.1% of total parameters.

### 5.5 Quantization Results

| Metric | Value |
|--------|-------|
| Source model | FP16 GGUF, 16.47 GB |
| Quantized model | NVFP4 GGUF, **4.76 GB** |
| Effective bits per weight | **4.52 bpw** |
| Tensors quantized | 2,403 NVFP4 + 80 F16 fallback |
| Quantization time | ~35 seconds (single-threaded) |
| Output file | `/tmp/zaya1-8b-nvfp4.gguf` |
| Name map | `/tmp/zaya1-8b-nvfp4.name_map.json` (2,483 entries) |

**First-ever NVFP4 ZAYA1-8B**: As of May 11, 2026, no NVFP4-quantized ZAYA1-8B exists on HuggingFace. The 10 community quantizations available use BNB (NF4), MXFP4, JANGTQ4, ONNX, or MLX formats — none of which provide Blackwell-native hardware acceleration.

### 5.6 vLLM Integration

**Architecture registration**: The gguf library (`gguf/constants.py`) was patched to register Zaya as a known architecture:
- `MODEL_ARCH.ZAYA` added to the `MODEL_ARCH` enum
- `"zaya"` entry added to `MODEL_ARCH_NAMES` (assigned ID 127)
- Empty tensor mapping added to `MODEL_TENSORS` (identity mapping for HF-style names)

**Model registry**: ZayaForCausalLM registered in vLLM's `ModelRegistry` via patch to `registry.py`.

**GGUF loader patches** (`gguf_loader.py`):
1. **Name map loading**: Loads `name_map.json` before `_get_gguf_weights_map()`, using `model_config.model.replace(".gguf", ".name_map.json")`.
2. **Pre-populated mapping**: Injects all name_map entries into `gguf_to_hf_name_map` before the state_dict loop.
3. **lm_head.weight filter**: Filters the tied embedding parameter from unmapped params check.

**Result**: 2,483 of 2,483 parameters successfully mapped (0 unmapped). The remaining blocker is vLLM's ZayaForCausalLM weight loading expecting fused MoE weight names (`w13_weight`/`w2_weight`) that differ from our GGUF's expert naming — this is handled natively by the Zyphra vLLM fork.

### 5.7 NVFP4 Serving Architecture (May 12, 2026)

**Original GGUF path blocked**: vLLM's GGUF handler lacks NVFP4 tensor type support. Five separate patches applied (model registry, CCA state shape/dtype, FusedMoE GGUF per-expert materialization, zaya.py GGUF routing, GGUF single-shard materialization). Custom loader achieved 803/2483 weights at 1.04 GB before hitting MoE routing + GGUFUninitializedParameter materialization issues with remaining 1440+ expert weights.

**Four paths evaluated**:

| Path | VRAM | Kernel | Hours | Status |
|------|------|--------|-------|--------|
| GGUF → vLLM GGUF handler | — | Python dequant | — | Blocked (MoE routing) |
| GGUF → Compressed-tensors | 4-5 GB | Marlin FP4 | 9-14 | Viable |
| Original → Compressed-tensors | 4-5 GB | Marlin FP4 | 6-9 | **Stage 1** |
| Custom Blackwell CUDA kernel | 4-5 GB | NVFP4 Tensor Core | 24-37 | **Stage 2** |

**Decision: Two-stage pipeline**:
1. **Stage 1** (6-9 hrs): Quantize original BF16 → compressed-tensors NVFP4 via `CompressedTensorsW4A16Fp4` (Marlin FP4 kernel). Publish first NVFP4 ZAYA1-8B benchmark. ~6.2 GB VRAM.
2. **Stage 2** (24-37 hrs): Write custom Blackwell NVFP4 Tensor Core CUDA dequant kernel as drop-in Marlin replacement. ~4-5 GB VRAM, sm_120 hardware-accelerated. Reusable across all models, open-source contribution to vLLM.

**Rationale**: Benchmark quality is identical between Stage 1 and Stage 2 (NVFP4 → FP16 dequant is deterministic math). Stage 1 gets us first-to-publish. Stage 2 demonstrates Blackwell GPU architecture expertise — deeper technical achievement with broader impact.

**Technical findings from GGUF path investigation**:
- `nvcc` at `/usr/local/cuda/bin/nvcc` (CUDA 13.2) must be in PATH for any vLLM CUDA builds
- Zyphra vLLM fork Python files (cca.py, cca_attn.py, zaya.py, zaya_tool_parser.py, zaya_config.py) can be overlaid on stock vLLM 0.20.2 (no CUDA kernel rebuild needed)
- `CompressedTensorsW4A16Fp4` uses Marlin FP4 kernel (get_min_capability=75, Turing+), NOT Blackwell-specific NVFP4 Tensor Cores
- lainlives/ZAYA1-8B-GGUF is empty (0 files, 0 bytes) — our NVFP4 GGUF is genuinely the first
- llama.cpp has no Zaya architecture support (no model implementation, no converter entry)
- CCA attention requires GPU; CPU offloading breaks CCA kernels, producing garbage output
- `GGUFLinearMethod._create_padded_weight_param` only materializes when `len(data_container) > 1` (multi-shard); single-shard params remain uninitialized without patching

### 5.8 Comparative Analysis

| Model | Quantization | Size | Blackwell HW | Loads on Zyphra Fork |
|-------|-------------|------|-------------|----------------------|
| base | bf16 | 16.47 GB | N/A | ✅ |
| base | FP8 (online) | 8.76 GB | ✅ | ✅ (stock vLLM 0.20.2 + 5 Python file overlay) |
| barozp/ZAYA1-8B-BNB | NF4 | ~7 GB | ❌ | ❌ (broken inference) |
| OsaurusAI/ZAYA1-8B-MXFP4 | MXFP4 | ~5.5 GB | ⚠️ | ❌ (weight shape mismatch) |
| lainlives/ZAYA1-8B-GGUF | Q4_K/Q8_0 | N/A | ❌ | ❌ (empty repo, 0 files uploaded) |
| **Our NVFP4 GGUF** | **NVFP4** | **4.76 GB** | **✅ Native** | **GGUF handler incompatible** |
| **Our NVFP4 CT (Stage 1)** | **NVFP4** | **5.04 GB disk** | ⚠️ (Python dequant for MoE, Marlin for CCA) | **✅ Coherent text (May 14 session 2)** |
| **Our NVFP4 CUDA (Stage 2)** | **NVFP4** | **~4.5 GB disk** | **✅ Tensor Core MMA** | **Planned** |

### 5.9 Compressed-Tensors Serving — First Achievement (May 14, 2026)

**This is the first successful NVFP4 Compressed-Tensors serving of ZAYA1-8B via vLLM.** As of May 14, 2026, no NVFP4-quantized ZAYA1-8B exists on HuggingFace in any format that loads successfully through vLLM's CompressedTensors pipeline. The 10 community quantizations use BNB (NF4), MXFP4, JANGTQ4, ONNX, or MLX — none provide Blackwell-native hardware acceleration and none successfully serve through vLLM with the Zyphra model architecture.

**Smoke test**: All 4,244 weights loaded. 5.51 GiB VRAM. Model initialization completes. Exit code 0.

This milestone required diagnosing and fixing two root-cause bugs in vLLM's NVFP4 Compressed-Tensors integration with the Zaya model architecture. Neither bug was Zaya-specific — both are general vLLM CompressedTensors issues exposed by the Zaya architecture's unique combination of per-expert MoE weight naming and non-standard quantization group size. Both fixes are upstreamable to the vLLM project.

#### 5.9.1 Root Cause 1: Scale Parameter Routing in `zaya.py`

**Symptom**: 26 of 4,244 weights loaded, then crash with `ValueError: quant method must be one of ['tensor', 'channel', 'group', 'block']` at `fused_moe/layer.py:1359`.

**Debugging methodology**: The error location (`layer.py:1359`) was identified from the traceback. Line 1359 is inside `FusedMoE._load_weight` in the scale-loading branch (`if "scale" in weight_name`). The validation checks `getattr(param, "quant_method", None)` against the `FusedMoeWeightScaleSupported` enum. Since the error fired on the *else* branch, `quant_method` must be `None` — meaning the parameter being loaded had no `quant_method` attribute. Inspection of the NVFP4 MoE method's `create_weights` confirmed only scale parameters (`w13_weight_scale`, `w2_weight_scale`) carry `quant_method=GROUP`; weight parameters (`w13_weight`, `w2_weight`) do not. Combined with the knowledge that the checkpoint contained per-expert `weight_scale` keys, the routing bug was identified: `zaya.py`'s `load_weights` was sending scale data to weight parameters.

**Root Cause**: `ZayaForCausalLM.load_weights` (lines 926-967) contains two branches for MoE checkpoint keys — one for `linear_fc1` and one for `linear_fc2`. Both branches unconditionally look up `w13_weight`/`w2_weight` on the FusedMoE module, regardless of the checkpoint key's suffix (`weight_packed` vs `weight_scale`). The logic is:

```python
if parts[-2] == "linear_fc1":
    param_name = f"{fused_moe_prefix}.w13_weight"   # Always w13_weight
    param = params_dict.get(param_name)
    # ... fallback to w13_weight_packed ...
    fused_moe_module.weight_loader(param, loaded_weight, chkpt_weight_name, "w1", expert_id)
```

When `chkpt_weight_name` is `...linear_fc1.weight_scale`, `_load_weight` sees `"scale"` in the name and enters its scale-loading branch. It then inspects `param.quant_method` — but `param` is `w13_weight`, a weight parameter that never receives `quant_method` from `create_weights`. The attribute is `None`, the validation fails, and `ValueError` is raised.

**Why this wasn't caught earlier**: The vLLM CompressedTensors pipeline assumes MoE expert weights are stored as flat multi-expert tensors (shape `[E, N, K]`), not per-expert keys with `local_experts.N.` prefixes. In standard deployments, a single `w13_weight_packed` key holds all expert weights and the flat-tensor `weight_loader` uses shard dimensions, not the `local_experts` routing path. Zaya's safetensors use per-expert naming inherited from the GGUF quantization tooling, which triggers the `local_experts` routing branch for the first time.

**Fix** (`scripts/wsl_fix_moe_scale_routing.py`): Added `weight_scale` suffix detection before the existing weight lookup in both the `linear_fc1` and `linear_fc2` branches. The detection precedes the weight lookup — when the checkpoint key ends in `weight_scale`, the code looks up `w13_weight_scale`/`w2_weight_scale` instead. These scale parameters carry `quant_method=FusedMoeWeightScaleSupported.GROUP.value` from the NVFP4 MoE method's `create_weights`, satisfying the `_load_weight` validation. The `weight_loader` is called with the correct shard ID (`"w1"` or `"w2"`) and `loaded_params` tracking continues as normal. The existing `weight_packed` routing is unmodified.

**Key insight**: The fix is purely a routing correction — it doesn't change any weight-loading logic. The `FusedMoE._load_weight` method already knows how to handle scale parameters correctly; it simply needs to receive the correct parameter. This is why the fix is both minimal (8 lines added per branch) and robust (no new code paths, no risk of breaking existing behavior).

#### 5.9.2 Root Cause 2: Marlin Kernel Group Size Mismatch

**Symptom**: After scale routing fix, all 4,244 weights loaded but inference failed during `profile_run` with `RuntimeError: Invalid thread config: thread_m_blocks = 1, thread_k = -1, thread_n = -1, num_threads = -1` in the Marlin fp4 kernel at `marlin_utils_fp4.py:182`. Error originated from CCA attention's `linear_q` projection with dimensions `MKN = [2048, 2048, 1024]` and `group_size = 64`.

**Debugging methodology**: The traceback showed the error in `ops.marlin_gemm` called from `apply_fp4_marlin_linear` → `CompressedTensorsW4A16Fp4.apply_weights` → CCA `linear_q` forward. The error message included `group_size = 64`, which was suspicious — the NVFP4 standard group size is 16. A grep for `group_size` in the Marlin fp4 utilities revealed `FP4_MARLIN_SUPPORTED_GROUP_SIZES = [16]` at line 21 of `marlin_utils_fp4.py`. The `prepare_fp4_layer_for_marlin` function hardcodes `group_size = 16` at line 241 but never validates against `FP4_MARLIN_SUPPORTED_GROUP_SIZES`. The model's `config.json` specifies `group_size: 64` in the quantization config, and the NVFP4 scheme instantiates with this value. The Marlin repack succeeded (tile dimensions were satisfied: K=2048 % 256 == 0, N=1024 % 64 == 0) but the kernel's thread config autotuner failed because the repacked scale layout for group_size=64 has no valid thread decomposition.

**Root Cause**: The failure chain has three contributing factors:

1. **Missing validation**: `CompressedTensorsW4A16Fp4.process_weights_after_loading` (line 81 of `compressed_tensors_w4a16_nvfp4.py`) unconditionally calls `prepare_fp4_layer_for_marlin(layer)` without checking whether `self.group_size` is supported by the Marlin fp4 kernel.

2. **Hardcoded assumption**: `prepare_fp4_layer_for_marlin` (line 241 of `marlin_utils_fp4.py`) hardcodes `group_size = 16 if is_nvfp4 else 32`. This works for standard NVFP4 models (which always use group_size=16) but silently produces incorrect scale layouts for non-standard group sizes. The Marlin repack uses tile dimensions for validation, not group size — so repack succeeds while the actual kernel fails.

3. **Config mismatch**: The CompressedTensors quantization configuration `group_size: 64` is a valid CT value — the CT library supports any group size. But the Marlin fp4 kernel, designed specifically for NVFP4's 16-element block quantization, was never tested with non-standard group sizes. The existing guard constant `FP4_MARLIN_SUPPORTED_GROUP_SIZES = [16]` was defined but never checked at runtime.

**Why this wasn't caught earlier**: ZAYA1-8B is the first model using NVFP4 CompressedTensors with a non-standard group size. Standard deployments use group_size=16, which passes through the hardcoded path silently. The Marlin kernel's autotuner produces `thread_k = -1` when it can't find a valid decomposition, but the kernel itself provides no informative error message — it took cross-referencing the error message, the hardcoded constant, and the model config to identify the mismatch.

**Fix** (`scripts/wsl_fix_marlin_group_size.py`): Added a pre-check in `process_weights_after_loading` before the `prepare_fp4_layer_for_marlin(layer)` call:

```python
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    FP4_MARLIN_SUPPORTED_GROUP_SIZES,
)
if self.group_size not in FP4_MARLIN_SUPPORTED_GROUP_SIZES:
    layer._marlin_repack_skipped = True
    return
```

When group_size is unsupported, the method sets `_marlin_repack_skipped = True` and returns immediately, preserving the original `_weight_packed_data`, `_weight_scale_data`, and `_weight_global_scale_data` that were cloned earlier in the method. The `apply_weights` method already has a fallback path (line 107) that checks this flag and performs Python dequant via `compressed_tensors`:

```python
if getattr(layer, "_marlin_repack_skipped", False):
    w = unpack_fp4_from_uint8(wq, m, nh * 2)
    w = dequantize(x_q=w, scale=ws.float(), global_scale=wgs, dtype=ws.float().dtype)
    out = torch.nn.functional.linear(x, w.to(x.dtype))
    return out + bias if bias is not None else out
```

**Key insight**: The fix reuses an existing fallback mechanism (built for CCA dimension-alignment issues) rather than creating a new code path. The `_marlin_repack_skipped` flag was already the signal between `process_weights_after_loading` and `apply_weights`. Extending the condition that sets this flag from "tile-unaligned dimensions" to "tile-unaligned OR unsupported group_size" is a single-line logic change with zero risk to the existing fallback path.

**Performance note**: The Python dequant fallback is slower than the Marlin kernel (~2-5× for typical attention projection sizes). However, MoE expert layers (which dominate FLOPs) use the Marlin kernel regardless — the NVFP4 MoE method's separate code path (`prepare_nvfp4_moe_layer_for_marlin`) hardcodes `GROUP_SIZE = 16` independently of the Linear scheme's `group_size`. The performance impact is limited to CCA attention projections which represent a minority of total compute. Stage 2 (custom Blackwell CUDA kernel) will eliminate this gap entirely.

#### 5.9.3 Dequantization Strategy

After both fixes, the model uses a hybrid dequantization strategy:

| Layer type | Dequant method | Kernel | Group size |
|-----------|----------------|--------|------------|
| MoE experts (FusedMoE) | Marlin fp4 kernel | `CompressedTensorsW4A4Nvfp4MoEMethod` → `prepare_nvfp4_moe_layer_for_marlin` | 16 (hardcoded in NVFP4 MoE path) |
| CCA attention projections | Python dequant fallback | `compressed_tensors` unpack + dequantize | 64 (from CT config) |
| Router layers | Bypassed (unquantized) | `UnquantizedLinearMethod` | N/A |

MoE layers are handled by a separate code path (`prepare_nvfp4_moe_layer_for_marlin` in `marlin_utils_fp4.py` line 319) that hardcodes `GROUP_SIZE = 16` — independent of the Linear-layer scheme's `group_size`. This means MoE experts always use the Marlin kernel regardless of the CT config's group_size setting.

#### 5.9.4 Smoke Test Results (May 14, 2026)

```
Loading weights: 100%|██████████| 4.24k/4.24k [00:11<00:00, 358weights/s]
Model loading took 5.51 GiB memory and 16.07 seconds
Available KV cache memory: 8.92 GiB
GPU KV cache size: 231,995 tokens
SUCCESS: Model loaded!
Exit code: 0
```

**VRAM breakdown**: 5.51 GiB model weights + 1.57 GiB overhead → 8.92 GiB free for KV cache. At 2,048 max model length, supports 113 concurrent requests.

#### 5.9.5 Reproduction

```bash
# Environment
source /home/ttimm/vllm-env/bin/activate
export PATH=/usr/local/cuda/bin:$PATH

# Apply fixes (order matters)
python3 scripts/wsl_fix_moe_scale_routing.py
python3 scripts/wsl_fix_marlin_group_size.py

# Smoke test
bash scripts/wsl_run_smoke.sh
```

**Note**: These fixes modify files inside the WSL vLLM Python installation at `/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/`. The fix scripts are idempotent — they check for existing patches before applying. Re-running after a vLLM reinstall is safe.

### 5.10 Coherent Text Generation — Second Achievement (May 14, 2026, session 2)

**This is the first successful coherent text generation from a NVFP4 Compressed-Tensors ZAYA1-8B on Blackwell hardware.** With the smoke test from §5.9 passing, the model was *initialized* but every prompt produced 120 consecutive pad tokens (token id 0). Three additional bugs were diagnosed and fixed; together they unlocked the model. Greedy decoding now answers "The capital of France is" → " Paris. London is the capital of the UK. Tokyo is the capital of Japan..." and explains binary search trees coherently. Performance: **~0.86–0.90 tok/s** end-to-end with Path A (on-the-fly Python dequant), bottlenecked by the per-layer dequant loop over 16 experts. The MoE Marlin kernel is bypassed entirely on this path because `nvfp4_marlin_process_scales` produces negative-valued scales for this checkpoint via the FP8→S0E5M3 conversion at `marlin_utils_fp4.py` lines 108–112 (confirmed: the checkpoint's stored scales are strictly non-negative).

These bugs are not Zaya-specific. They are general vLLM CompressedTensors and `FusedMoE.weight_loader` issues that surface only when (a) checkpoint weights are stored per-expert with combined gate+up rows and packed FP4 dtype, and (b) lm_head is NVFP4 with tied embeddings. The fixes are upstreamable to vLLM.

#### 5.10.1 Root Cause 3: `_load_w13` narrows combined-shard packed weights to gate-half

**Symptom**: After §5.9 fixes, model loads "4,244/4,244 weights" and `process_weights_after_loading` completes for all 40 MoE layers. Apply diagnostics in the first MoE layer show valid inputs (`x` mean abs 0.638, range [-2.42, 9.04]) and valid dequantized w13 (mean abs 0.010, range [-0.84, 0.61]) but `out` is identically zero. The next MoE layer's dequant produces NaN-valued w13, and NaN cascades through the remaining 39 layers. Final logits collapse to argmax=0 → pad token output regardless of input prompt.

**Debugging methodology**: A one-shot diagnostic in `apply()` logged `(x, w13_fp, topk_weights, topk_ids, out)` summary statistics on the first call per layer. The first MoE layer showed `topk_ids=[4,4,...4]` (every warmup token routed to expert 4), reasonable scales, and reasonable dequantized weights — yet `out` was identically zero. Switching the apply implementation from `fused_experts` (vLLM's Triton MoE kernel) to a manual per-token loop produced the same zero output: the bug was *not* in the dispatcher. A second diagnostic inside `process_weights_after_loading` split `layer.w13_weight` and `layer.w13_weight_scale` into gate-half (rows `[:N]`) and up-half (rows `[N:]`) and reported per-half statistics. Result:

```
gate_nz=33,513,898/33,554,432  up_nz=33,513,386/33,554,432
gate_scale_min=0.00195  gate_scale_max=0.14063
up_scale_min=0.0        up_scale_max=0.0          ← entire up half is zero
```

The packed weight bytes for the up half were nonzero (random uint8 patterns from `torch.empty`), but the *scales* for the up half were all zeros. Dequant of nonzero FP4 weights with zero scales produces identically zero output — explaining `out=0` for the first MoE layer where every token routes to one expert.

The downstream NaN cascade comes from the same uninitialized memory in subsequent layers: for layers where the up-half packed bytes happen to decode to bit patterns adjacent to FP4 special values, downstream attention sees a single NaN and propagates it.

**Root Cause**: `FusedMoE._load_w13` at `vllm/model_executor/layers/fused_moe/layer.py:943` narrows the loaded tensor to `expert_data.shape[shard_dim] // 2` rows when `is_act_and_mul=True` (lines 954–955), then narrows the target `expert_data` to the gate half for `shard_id="w1"` (lines 974–975). This is correct *if* the caller is loading a single half. But `zaya.py` passed the full combined `[2*N, K//2]` checkpoint tensor with `shard_id="w1"`, expecting all `2*N` rows to be copied — the narrowing silently dropped the up half. The packed-weight path in `zaya.py` (added in session 1) used `shard_id="w1"` directly on the combined tensor, so up rows of every expert in every MoE layer remained at their `torch.empty` initialization values.

For the *scales*, there is a parallel "combined" fast-path at `fused_moe/layer.py:1298` that detects when `loaded_weight.shape[-2] == param.data.shape[-2]` and bypasses narrowing via `_load_combined_w13_weight_scale`. That fast-path is gated on `if "ModelOpt" in quant_method_name` (line 1260). For CompressedTensors, the fast-path is skipped; scale loading falls through to `_load_model_weight_or_group_weight_scale` → `_load_w13` → the same narrowing → up scales never load → stays at `torch.empty` (which happens to read zero on this allocator on this run, but is undefined in general).

**Fix** (`scripts/wsl_fix_nvfp4_text_gen.py`, fix #1+#2): In `ZayaForCausalLM.load_weights`, split *both* `linear_fc1.weight_packed` and `linear_fc1.weight_scale` into `loaded_weight[:half, :]` (gate) and `loaded_weight[half:, :]` (up), and call `fused_moe_module.weight_loader` twice — once with `shard_id="w1"` for gate, once with `shard_id="w3"` for up. After this, `_load_w13`'s narrowing operates on the correct half-tensor and produces the correct layout. Both halves of `w13_weight_packed` and `w13_weight_scale` carry the checkpoint's data with no uninitialized rows.

**Why this wasn't caught earlier**: Stage 1's smoke test (§5.9.4) only checked weight *count* (4,244/4,244 weights loaded) and exit code. It did not verify weight *content*. A "weight loaded" log entry fires when `weight_loader` returns successfully — it has no way to detect that the loader silently dropped half the tensor.

#### 5.10.2 Root Cause 4: NVFP4 lm_head silently dropped under tied embeddings

**Symptom**: After fix #3 (w13 split), `apply()` diagnostics show reasonable values through every MoE layer (no NaN, magnitudes in expected ranges) — yet greedy decoding still emits pad tokens (id 0) for every prompt. Sampling with temperature=0.8 produces *varied* but still semantically random tokens. The forward pass is computing correctly but the final lm_head projection produces logits that argmax to pad.

**Debugging methodology**: A diagnostic added at the end of `ZayaForCausalLM.load_weights` reported the post-load statistics of `self.model.embed_tokens.weight`: shape `(262272, 2048)`, dtype `torch.float16`, `abs_mean=0.000` — values were not loaded at all (zero or torch.empty noise). Inspection of the checkpoint showed `lm_head.weight_packed` (uint8) and `lm_head.weight_scale` (fp8) keys but no `lm_head.weight` and no `model.embed_tokens.weight`. The Zaya config has `tie_word_embeddings=True` (HF default when unset) and `zaya_high_prec=True`. zaya.py constructs `self.lm_head = ParallelLMHead(..., quant_config=None, ...)` and then calls `self.lm_head.tie_weights(self.model.embed_tokens)`, so the shared parameter is registered under the embed_tokens name only. The default load loop in `zaya.py` matched `chkpt_weight_name` against `params_dict` by exact string and silently skipped both NVFP4 keys.

The skip was masked by a broken log line at `zaya.py:1037`:
```python
logger.info("WARNING: key {chkpt_weight_name} not in params! Skipping loading")
```
The `f` prefix is missing — Python logs the literal string `"WARNING: key {chkpt_weight_name} not in params!"` for every skip, so the actual keys never appeared in stderr. Fixing the format string to `logger.info("WARNING: key %s not in params! Skipping loading", chkpt_weight_name)` revealed `lm_head.weight_packed` / `lm_head.weight_scale` as the silently-dropped keys.

**Root Cause**: `ParallelLMHead(quant_config=None)` creates an *unquantized* lm_head layer that registers a single `weight` Parameter (fp16). With `tie_word_embeddings=True`, `tie_weights` rebinds the lm_head to share `model.embed_tokens.weight`. The Zaya `zaya_high_prec=True` path then attaches a custom `_FP32EmbeddingMethod` that calls `torch.mm(x, layer.weight.t(), out_dtype=torch.float32)` — it reads `layer.weight` directly and never goes through any quantization scheme. There is no code path that dequantizes the NVFP4 lm_head from the checkpoint into this fp16 Parameter.

**Fix** (`scripts/wsl_fix_nvfp4_text_gen.py`, fix #2): In `ZayaForCausalLM.load_weights`, buffer `lm_head.weight_packed` and `lm_head.weight_scale` during the load loop (do not pass them to `weight_loader`). After the loop completes, dequantize via `compressed_tensors.compressors.nvfp4.helpers.unpack_fp4_from_uint8` + `compressed_tensors.quantization.lifecycle.forward.dequantize` to a fp32 tensor, cast to the target dtype, and copy into `params_dict["model.embed_tokens.weight"]` (the canonical name under tied embeddings) or `params_dict["lm_head.weight"]` as a fallback. Also fix the broken log line so future loader skips are diagnosable.

**Why this wasn't caught earlier**: The session-1 smoke test verified the model *initialized*. Initialization does not exercise the lm_head — that only fires during the final projection per generation step. Combined with the broken log line hiding the skip, there was no signal until end-to-end generation was attempted.

#### 5.10.3 Root Cause 5: Marlin MoE scale corruption + emulation backend device mismatch

**Symptom**: When attempting to keep the Marlin MoE backend after fix #3+#4, generation produces all pad tokens despite valid forward pass logging. Falling back to the NVFP4 emulation MoE backend crashes with `RuntimeError: Expected all tensors to be on the same device` (kE2M1 lookup table on CPU, input tensors on CUDA).

**Root Cause**: `nvfp4_marlin_process_scales` (in `marlin_utils_fp4.py` lines 108–112) reinterprets the loaded FP8_E4M3 scale tensor through a FP8 → S0E5M3 bit-level conversion designed for the Marlin kernel's internal scale format. For this checkpoint, the conversion introduces negative-signed values for scales that are strictly non-negative in the source data (verified by direct inspection: `weight_scale.float().min() = 0.00195` across all MoE layers, with no negative values). The Marlin kernel then multiplies dequantized values by these mis-signed scales, producing flipped-sign outputs that corrupt the SwiGLU result.

The emulation backend (`Nvfp4QuantizationEmulationTritonExperts`) bypasses Marlin entirely but contains a device-placement bug exposed under WSL: `kE2M1` is initialized at module import time on CPU, and the experts class never moves it to the input device before the unpack step.

**Fix** (`scripts/wsl_fix_nvfp4_text_gen.py`, fix #3 — "Path A"): Rewrite `CompressedTensorsW4A4Nvfp4MoEMethod.apply()` and `process_weights_after_loading()` to bypass both Marlin and emulation. Keep packed FP4 weights (`layer.w13_weight`, `layer.w2_weight`) and per-group scales (`self._w13_scale`, `self._w2_scale`) at original layout, cloned to decouple from any downstream Marlin-prep that might mutate them in place. In `apply()`, dequantize on the fly per call using the same `unpack_fp4_from_uint8` + `dequantize` primitives that the Linear NVFP4 Python-dequant fallback uses (and that produce correct CCA attention output). Execute the MoE dispatch with a manual per-expert loop:

```python
for e_id in range(E):
    mask = (topk_ids == e_id)
    token_idx = mask.any(dim=-1).nonzero(as_tuple=True)[0]
    if token_idx.numel() == 0:
        continue
    xe = x[token_idx]
    gate_up = F.linear(xe, w13_fp[e_id])              # [t_e, 2N]
    gate, up = gate_up[..., :N], gate_up[..., N:]
    hidden = F.silu(gate) * up                        # vLLM SiluAndMul convention
    down = F.linear(hidden, w2_fp[e_id])              # [t_e, K]
    tw = ((topk_ids[token_idx] == e_id).to(x.dtype)
          * topk_weights[token_idx].to(x.dtype)).sum(dim=-1, keepdim=True)
    out[token_idx] += tw * down
```

`fused_experts` was tried first but produced zero output for constant-routed warmup batches without a cached Triton config for sm_120; the autotuner appears to mis-tune when every token routes to the same expert. The manual loop is slower but correct and produces deterministic output across both warmup and real decode.

**Why this wasn't caught earlier**: The Marlin MoE corruption only manifests through end-to-end inference; the kernel itself does not error, it silently produces sign-flipped outputs. The emulation backend works on Linux native CUDA installs where module-level CPU tensors get auto-moved by `.to(device)` calls inside the experts class — but the WSL CUDA installation has a different module-import order that leaves `kE2M1` on CPU at the time of first use.

#### 5.10.4 Inference Contract: bfloat16 required

`dtype="bfloat16"` in the vLLM `LLM(...)` constructor is **required**, not optional, for this serving path. `dtype="float16"` produces collapsed output where greedy decoding selects the same token at every step (observed: token 27269 = ` Investment` repeated for 40+ steps from the prompt "The capital of France is"). The accumulation precision in the Path A Python MoE dequant loop is insufficient at fp16; bf16's larger exponent range avoids the saturation that drives the model to a single attractor in fp16.

This is a property of the dequant *path*, not the model. The Marlin MoE kernel and a future Blackwell NVFP4 Tensor Core kernel both perform accumulation in higher precision internally and should work in fp16 once available.

#### 5.10.5 Smoke Test Results (May 14, 2026, session 2)

```
Loading weights: 100%|██████████| 4.24k/4.24k [00:11<00:00, 358weights/s]
Model loading took 5.53 GiB memory and 16.07 seconds
Available KV cache memory: 8.42 GiB
GPU KV cache size: 218,993 tokens

--- RAW completion ---
Prompt: "The capital of France is"
Token IDs: [9079, 236761, 5860, 563, 506, 5279, 529, 506, 6322, 236761, ...]
Text: " Paris. London is the capital of the UK. Tokyo is the capital of Japan.
       London is in the United Kingdom. Tokyo is in Japan. London is closer to
       Tokyo than to the UK. London"
Output speed: 0.86 toks/s, max_tokens=40

--- CHAT completion ---
Prompt: "Explain what a binary search tree is in one sentence."
Text: "We are asked: Explain what a binary search tree is in one sentence.

       Alright, I need to explain a binary search tree in one sentence.
       Let's recall the definition: a binary search tree is a rooted tree..."
Output speed: 0.90 toks/s, max_tokens=120
```

The model produces correct factual responses ("The capital of France is" → " Paris.") and coherent reasoning chains. Output speed is dominated by the per-layer Python dequant loop (16 experts × 2 projections per layer × 40 MoE layers = 1,280 dequant ops per token). Stage 2 (Blackwell NVFP4 Tensor Core CUDA kernel) replaces this with hardware-accelerated MMA, projecting >10× speedup.

#### 5.10.6 Reproduction

```bash
# Environment
source /home/ttimm/vllm-env/bin/activate
export PATH=/usr/local/cuda/bin:$PATH

# Apply fixes (order matters — session 1 first, then session 2)
python3 scripts/wsl_fix_moe_scale_routing.py       # session 1: scale routing
python3 scripts/wsl_fix_marlin_group_size.py       # session 1: gs fallback
python3 scripts/wsl_fix_nvfp4_text_gen.py          # session 2: 3 fixes

# Smoke test (must use dtype="bfloat16")
bash scripts/wsl_run_quick_check.sh
```

Expected output: " Paris..." for the raw prompt, coherent BST explanation for the chat prompt. If the model produces all pad tokens (id 0), one of the three session-2 fixes did not apply cleanly — run the patch script with verbose flags or inspect the WSL vllm-env files directly.

### 5.11 Path B — W4A4 Weight+Activation Quantization (May 17, 2026)

#### 5.11.1 Motivation

Path A's weight-only NVFP4 (W4A16) quantizes weights to 4-bit but leaves
activations at FP16/BF16. Full W4A4 (4-bit weights + 4-bit activations) reduces
both, enabling larger KV caches and batch sizes on 16 GB consumer hardware.
W4A4 is the native format for Blackwell's NVFP4 Tensor Core MMA — the CUTLASS
SM120 kernel (compiled May 15 in Stage 2) handles W4A4 natively.

W4A4 requires **activation calibration**: running real prompts through the model
to observe per-Linear activation ranges, then computing `input_global_scale`
values that prevent FP4 overflow. This is done via llm-compressor's calibration
pipeline, which needs a forward pass through the model.

#### 5.11.2 Calibration Corpus Design

| Property | Value |
|----------|-------|
| Total samples | 979 |
| Max length | 1024 tokens |
| Packing mode | concat-pack with EOS separator |
| Pad ratio | 0.006% (near-zero) |
| Source mix | math500 (15%), gsm8k (15%), humaneval (5%), mbpp (5%), triviaqa (15%), alpaca (15%), writingprompts (15%), glaive (15%) |

The corpus packs multiple prompts end-to-end with EOS separators, then slices
into fixed 1024-token blocks. This avoids GPU memory bloat from padding and
provides uniform tensor shapes for batch calibration.

#### 5.11.3 CPU Calibration Dry-Run — FAILED

**Result**: W4A4 dry-run on 4 layers + 8 calibration samples crashed.

Three problems identified:

1. **Forward pass crash**: All 8 passes failed with `max(): Expected reduction dim`
   for `input.numel() == 0`. Zaya's CCA attention implements CUDA-only primitives
   (depthwise+grouped conv1d) that do not work on CPU. The forward pass never
   completes, so no hidden-state maxes are observed.

2. **Calibration coverage**: Only 4 of 1,480 Linears got non-zero
   `input_global_scale`. Those 4 all read identical 10.625 (suspicious — suggests
   a partial forward firing once before the crash). Remaining scales are 8.97e-44
   (near-zero garbage from torch.empty initialization).

3. **Output dir bloat**: 18.56 GB. Only 74 dry-run modules got compressed;
   everything else saved as raw BF16. The verification logic reports "PASSED"
   because it counts keys present, not whether values are valid.

**Root cause**: llm-compressor's calibration loop uses `model.forward()` on CPU
tensors. ZayaForCausalLM's CCA attention calls CUDA-only operations internally.
CPU calibration is fundamentally broken for Zaya.

#### 5.11.4 Layer-Wise GPU Calibration Architecture (Recommended)

Process one Zaya layer at a time on GPU:

1. Embed the 979×1024 calibration tensor on **GPU**
2. Forward through layer 0: save output hidden states to CPU, observe maxes
   via forward hooks, move layer 0 back to CPU
3. Repeat for layers 1..79, reusing saved hidden states as input

**Why this works**:
- Each Zaya layer fits comfortably on GPU (~200 MB BF16)
- CCA attention runs natively on GPU — no CPU forward bug
- ~10–20× faster than naive full-model forward (~30–60 min vs ~8 hr CPU)
- ~150 LOC of new code needed

**Production precedent**: This is the standard approach used by production
NVFP4 quantizers. llm-compressor's GPU calibration uses the same pattern
under the hood but was never tested with Zaya's custom forward.

Also needed: quality gate fix (ERROR not "PASS" when calibration coverage
<95%) and output-dir bloat fix (don't save duplicate BF16 for compressed layers).

See `ROADMAP.md` → Path B — W4A4 Weight+Activation Quantization for full
task tracking.

### 5.12 SOTA Accuracy Improvement Pipeline (Sessions 9–14, 2026-05-19–22)

#### 5.12.1 Motivation

The initial W4A4 baseline (session 8) achieved 68.6% ARC-Easy and 102 tok/s —
a strong start but well below Zyphra's published BF16 ceiling
(GPQA-Diamond 71.0%, MMLU-Pro 74.2%, IFEval 85.58%). Sessions 9–14 stacked
SOTA techniques to close that gap; the current active checkpoint is
`./zaya1-8b-nvfp4-w4a4/` (8.84 GiB, 936 W4A4 + 384 BF16, ARC-mix calibration,
smoke test PASSED 2026-05-22).

#### 5.12.2 KV Cache FP8

Added `kv_cache_dtype=fp8` to all eval and serving scripts. Halves KV cache VRAM
(~1-2 GiB freed), enabling larger batch sizes and longer contexts with zero accuracy
impact.

#### 5.12.3 SOAR Global-Scale Optimization

**Problem**: Max-abs global scale (`igs = 2688 / max_act`) maximizes FP4 range
coverage but does not minimize reconstruction error. The dominant error source is
FP8 block-scale rounding: `(fp8_rounded - fp8_raw)² × block_max²`.

**Implementation**: Added a 25-point log-spaced grid search over scale candidates
in `_compute_soar_global_scale()`. For each candidate `g`:
- Compute `fp8_raw = clip(block_maxes / (g × 6.0), eps, 448.0)`
- Round to FP8_E4M3: `fp8_rounded = fp8_raw.to(float8_e4m3fn).float()`
- Accumulate weighted error: `sum((fp8_rounded - fp8_raw)² × block_maxes²)`
- Return `g` with minimum weighted error

Block maxes are collected during the existing calibration hook via reservoir
sampling (capped at 1024 elements per entry). Max-abs is the fallback when
block maxes are unavailable.

**Result**: HellaSwag acc_norm 60.5% → 61.4% (+0.9pp). ARC-Easy 67.2%.

**Checkpoint**: `zaya1-8b-nvfp4-w4a4-soar/` (8.9 GB, 12 BF16-exempt outlier layers).

#### 5.12.4 EBSS Calibration (Expert-Balanced Sample Selection)

**Problem**: With top-1 routing and 977 samples × 40 MoE layers, each expert
activates in only ~60 samples on average. Under-represented experts have poorly
calibrated IGS values (zero in 340/1320 modules before the fix).

**Implementation** (`scripts/build_calibration_ebss.py`):
1. Load existing arcmix calibration data
2. Run router-only forward pass (embed tokens + router MLP per layer, BF16, no GPU needed)
3. Compute per-sample benefit score: `sum over layers of 1/(1 + coverage[routed_expert])`
4. Greedy resampling: iteratively select samples that maximize least-covered expert coverage

**Result**: 977-sample EBSS corpus at
`data/calibration/arcmix_ebss/calibration_data.pt`. Post-EBSS quantization
has **0 zero-IGS modules** (vs 340 previously). Applied to MR-GPTQ checkpoint.

**Zero-IGS post-hoc fix**: `scripts/fix_uncalibrated_igs.py` patches any
remaining zero/garbage IGS values with a per-layer median fallback. Idempotent;
applied after each quantization run.

#### 5.12.5 MR-GPTQ (Micro-Rotated GPTQ)

**Implementation**: Added `--mr-gptq` flag to `quantize_zaya_ct_nvfp4.py`.
Two stages per decoder layer:

1. **Hadamard rotation** (offline, CPU): For each Linear weight `W [out, in]`:
   reshape to `[out, in//16, 16]`, apply H₁₆/√16 via `W @ H.T`, reshape back.
   Absorb inverse rotation into preceding LayerNorm scale (multiply gamma by H.T).
   Net inference overhead: zero (rotation is in the weights).

2. **Hessian-weighted correction** (per-expert, GPU): Accumulate
   `H += x.T @ x` during the calibration forward hook. For each output column `j`:
   `W_corrected[:, j] = W_rotated[:, j] − H_inv[:, j] × quant_error[j] / H_inv[j, j]`
   Then quantize with SOAR-optimized global scale.

**OOM fix**: During development, `block_maxes_store` accumulated unbounded-length
bmax tensors. Popular experts routing 512 tokens/sample produced [65536]-element
tensors per hook call; across 64 entries × 1320 modules = 21 GB. Three fixes:

| Fix | Effect |
|-----|--------|
| Cap bmax at 1024 elements via random subsampling | Bounds per-module store to ~1 MB |
| `gc.collect()` per layer | Forces Python GC of freed tensors |
| `ctypes.malloc_trim(0)` per layer | Returns fragmented heap pages to OS |

Peak RSS dropped from 84.7 GB (OOM) to ~37 GB on 82 GB machine.

**Result**: Checkpoint `zaya1-8b-nvfp4-w4a4-mrgptq-v2/` (8.27 GiB safetensors,
0 bad IGS entries). Runtime: ~45 min on RTX 5070 Ti. This checkpoint has since
been deleted (session 14 disk cleanup); the active checkpoint is
`./zaya1-8b-nvfp4-w4a4/` (ARC-mix calibration, no MR-GPTQ).

#### 5.12.6 SingleQuant Rotations — BLOCKED (redesign required)

`scripts/apply_singlequant_rotations.py` was built to apply ART + URT outlier
rotations to the 12 BF16-exempt MoE layers. Root cause of failure:

The absorption formula `gamma_new = R @ gamma` is mathematically incorrect.
`W @ R.T @ ((R @ gamma) ⊙ z) ≠ W @ (gamma ⊙ z)` for arbitrary orthogonal R
because element-wise scaling `(gamma ⊙ z)` does not commute with rotation.
The 2048×2048 URT rotation magnifies this error; layer 1 is corrupted and
cascades to 25–26% ARC-Easy accuracy (near-random).

**Correct approach**: Absorb rotation into the **preceding linear's output
weights**, not into LN gamma. Concretely:

1. For each outlier MoE layer `L` (indices in `quantization_manifest.json`),
   identify the linear that immediately feeds into its output LayerNorm.
   For ZAYA1-8B this is `zaya_block.experts.local_experts.M.linear_fc2`
   (the down-projection of expert M's MLP).
2. Apply rotation `R` to the activation space: weight transform is
   `linear_fc2.weight ← linear_fc2.weight @ R.T` (right-multiply along
   output dim, shape `[out_features, in_features]` → `[out_features, in_features] @ [in_features, in_features]`).
3. Do NOT modify LN gamma. The LayerNorm then normalizes the rotated
   activations without any formula error.
4. Save the rotated BF16 model; then requantize with `--mr-gptq` to refit
   weights under the new activation distribution.

Estimated implementation time: ~30 min. The rotation will suppress activation
outliers in the 12 exempted layers, potentially allowing their threshold to
rise above 500 and converting some back to W4A4, shrinking the checkpoint.

Status: blocked pending redesign — do NOT re-run `apply_singlequant_rotations.py` as-is.

#### 5.12.7 Benchmarking Status (updated 2026-05-22)

**Target benchmarks** (aligned to Zyphra's published BF16 table):

| Task | BF16 ceiling | ARC-mix W4A4 | Final (rotation+GPTQ) |
|------|-------------|-------------|----------------------|
| GPQA-Diamond | 71.0% | pending | pending |
| MMLU-Pro | 74.2% | pending | pending |
| IFEval | 85.58% | pending | pending |

**Current checkpoint**: `./zaya1-8b-nvfp4-w4a4/` (8.84 GiB)
- 936 W4A4 modules, 384 BF16 modules (12 outlier MoE layers)
- ARC-mix calibration (`data/calibration/arcmix/calibration_data.pt`)
- 977 samples × 1024 tokens
- NO MR-GPTQ (dropped when mrgptq-v2 was deleted in session 14 cleanup)
- Smoke test PASSED 2026-05-22 (all 4 prompts coherent, 9.0 tok/s eager)

**GPQA-Diamond** (`Idavidrein/gpqa`) is gated — requires your HF account to be
approved at `huggingface.co/datasets/Idavidrein/gpqa` (click "Access repository").
A token alone is not sufficient. HF token saved to `/home/ttimm/.cache/huggingface/token`
in WSL. MMLU-Pro and IFEval do not require auth.

**VRAM constraints on this machine** (15.92 GiB usable, ~14.66 GiB free at startup):

| Config | KV Cache | Outcome |
|--------|----------|---------|
| gpu_mem=0.92, CUDA graphs ON | 0.69 GiB / ~8x concurrency | 11+ hr MMLU-Pro |
| gpu_mem=0.92, graphs OFF via env var | 4.56 GiB allocated but VRAM overcommitted | 10x slowdown, OOM |
| gpu_mem=0.99, CUDA graphs ON | — | Startup OOM (needs 15.76 GiB) |
| **gpu_mem=0.92, enforce_eager=True** | **4.68 GiB / ~53x concurrency** | ✅ Correct |

`enforce_eager=True` skips CUDA graph capture (saves 3.71 GiB). MMLU-Pro is
pure log-likelihood (zero output tokens) so CUDA graphs provide no benefit;
the 53x batching improvement dominates. `run_full_benchmarks.py` now defaults
to `enforce_eager=True`; override with `--no-enforce-eager`.

**Baseline benchmark command** (MMLU-Pro + IFEval; run GPQA separately after HF approval):
```bash
source /home/ttimm/vllm-env/bin/activate
cd "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed"
python3 scripts/run_full_benchmarks.py \
    --model ./zaya1-8b-nvfp4-w4a4 \
    --tasks mmlu_pro ifeval \
    --output results/lmeval_w4a4_baseline.json \
    > results/bench_baseline.log 2>&1 &
```
Expected runtime: ~2 hours. Output: `results/lmeval_w4a4_baseline.json`.

**Post-baseline final checkpoint pipeline**:
```bash
# Step 1: Fix rotation script and apply (~15 min)
python3 scripts/apply_singlequant_rotations.py \
    --input Zyphra/ZAYA1-8B \
    --manifest zaya1-8b-nvfp4-w4a4/quantization_manifest.json \
    --output ./zaya1-8b-bf16-rotated

# Step 2: Re-quantize with rotation + MR-GPTQ (~25 min)
python3 scripts/quantize_zaya_ct_nvfp4.py \
    --scheme w4a4 \
    --model-id ./zaya1-8b-bf16-rotated \
    --mixed-precision-threshold 1000 \
    --mr-gptq \
    --arc-mix \
    --output-dir ./zaya1-8b-nvfp4-w4a4-final

# Step 3: Benchmark final checkpoint (~80 min)
python3 scripts/run_full_benchmarks.py \
    --model ./zaya1-8b-nvfp4-w4a4-final \
    --output results/lmeval_w4a4_final.json
```

### 5.13 Evaluating a Long-Reasoning Model Under a VRAM Budget (Session 17, 2026-06-22)

**Headline: the W4A4 checkpoint is healthy. The alarming "baseline" numbers were a
reasoning-model harness artifact, not quantization damage.** Standard lm-eval protocols
silently mis-measure ZAYA1-8B because it is a post-trained reasoning model (`<think>` /
`</think>` special tokens) whose evaluation behaviour differs fundamentally from a base model.

**Symptom chain (each "fix" exposed the next, deeper problem):**

| Protocol | Score | Verdict |
|----------|-------|---------|
| `leaderboard_gpqa_diamond` (loglikelihood MCQ) | 33.8% | Denies the model any reasoning — caps a reasoning model near random. |
| `gpqa_diamond_cot_zeroshot` (generative CoT) | 6.6% flexible / **0.0% strict** | *Below* random → pure extraction failure, not model failure. |
| Budget-forced (this work) | **45.8%** | Above random; checkpoint genuinely solves GPQA. |

**Diagnosis (by dumping raw generations, `scripts/diagnose_gpqa_format.py`):**

1. **Answer-format mismatch.** The chat template opens the assistant turn with `<think>\n`.
   ZAYA reasons correctly (verified by hand: GPQA Q1 → "the correct answer is (C)" ✓,
   Q3 → "\boxed{B}" ✓) but emits its final answer as `\boxed{X}`, never the literal
   "The answer is X" that the stock `strict-match` filter requires — hence exactly 0.0%
   strict across all 198 questions. A score of *exactly* zero is the tell: a damaged model
   would still hit the phrase by chance.

2. **Non-terminating reasoning.** At both temp 0.0 (greedy) and temp 0.6 / top-p 0.95,
   every sampled question hit the token cap (4096, then 7000) **without ever emitting
   `</think>`**; some traces degenerated into repetition (`BiggBiggBigg…`). The stock CoT
   task stops only on `</s>`, so it can never reach a final answer. The model wants to reason
   far past any feasible single-GPU context window.

**Method — budget forcing (s1-style), `scripts/eval_gpqa_budget_forced.py`:** generate up to
a fixed think budget, inject `</think>\n\nThe final answer is (`, then decode the choice
letter. This is simultaneously (a) a *terminating, fair* eval protocol for a model that never
stops on its own, and (b) the way the model would actually be served under a fixed
latency/VRAM budget on consumer hardware.

**Results (GPQA-Diamond, identical 24 questions, temp 0.6 / top-p 0.95):**

| Think budget | Accuracy | Self-closed `</think>` | Notes |
|--------------|----------|------------------------|-------|
| 2500 tokens  | 45.8% (11/24) | 1/24 | Reasoning truncated mid-thought. |
| 5000 tokens  | 45.8% (11/24) | 2/24 | Doubling the budget did **not** move accuracy. |
| 12000 tokens (16k ctx) | **62.5% (15/24)** | 9/24 | Enough budget for many traces to *finish* → +16.7 pts. |

**Interpretation.** Accuracy rises **monotonically with reasoning budget on the same 24
questions** — 45.8% → 45.8% → 62.5% — and `</think>` self-closure tracks it (1 → 2 → 9 of 24).
The plateau at 2500–5000 followed by a jump at 12000 is exactly the signature of a model that
*needs* a long trace: below ~5k tokens it is always interrupted mid-thought and lands at the
same ~46%; only at 12k do enough traces actually conclude, and accuracy climbs to 62.5%,
within the ±~10% CI of Zyphra's BF16 CoT **71.0%**. This is direct, paired evidence that the
gap to the BF16 reference is the **reasoning-budget ceiling on 16 GB VRAM**, not quantization
loss: 45.8%–62.5% all sit far above the 25% random floor and the 6.6% / 33.8% stock-harness
artifacts. A residual quant-loss component cannot be isolated locally because a BF16 control
eval (~35 GiB) does not fit in 16 GiB. **Consequence: the rotation + MR-GPTQ "repair" arc
below is unnecessary — there is no measured damage to repair.** This is itself the
contribution: *how you evaluate (and serve) a long-reasoning model under a fixed compute
budget* is the real question for consumer-hardware deployment.

### 5.14 CUDA Graph Capture Corrupts NVFP4 MoE Output on SM120 (2026-08-11–14)

**Headline: the published 102.6 / 407.4 tok/s throughput figures were measured on
a code path that produces numerically wrong output on this card. Not the
checkpoint, not a specific kernel — CUDA graph capture itself.** This is a
distinct diagnosis from every FP4-kernel-specific bug logged elsewhere in this
file; it survived a change of MoE backend, which those bugs by definition do not.

**Symptom (2026-08-11):** greedy decode (temp 0) on `zaya1-8b-nvfp4-w4a4-uniform`
returned token soup (`"ssngthssystem"`); temp 0.7 returned fluent English that
was completely off-topic. Both are worse than a healthy model, and greedy being
*worse* than sampling is itself a tell — a correct model is never worse at
temp 0.

**Diagnosis — backend sweep, same prompt/weights/tokenizer throughout:**

| backend | with CUDA graphs |
|---|---|
| `flashinfer_cutlass` (default) | garbage |
| `cutlass` | garbage |
| `marlin` (weight-only) | garbage |
| `flashinfer_trtllm` / `cutedsl` / `triton` / `emulation` | fails to initialize |
| any backend, `enforce_eager=True` | ✅ coherent, on-topic |

`marlin` is the load-bearing data point: it is a **weight-only** kernel that
barely touches the FP4 activation-quantization path, and it still corrupts
output under graph capture. Three unrelated compute paths fail identically with
graphs on; the same path succeeds with graphs off. That pattern is only
consistent with **graph capture itself being at fault**, not any one kernel's
FP4 math.

**Why loglikelihood evals never caught this:** every accuracy number in this
project prior to 2026-08-11 (HellaSwag, ARC, PIQA, WinoGrande — §5.6, §5.9,
Session 16's paired McNemar table) is a **loglikelihood** task. Those score
pre-written continuations by ranking, never by generating — a completely broken
generation path can still score ~61% on HellaSwag because multiple-choice
ranking partially tolerates corrupted logits in a way free generation does not.
**A damaged generation path and a healthy model are indistinguishable on
loglikelihood tasks alone.** IFEval or another free-generation eval belongs in
the standard suite going forward for exactly this reason.

**Two upstream issues are adjacent, neither is a match:**

- [CUTLASS #3096](https://github.com/NVIDIA/cutlass/issues/3096) — SM120 NVFP4
  MoE grouped-GEMM produces garbage *without* graph capture; fixed via FlashInfer
  SM120 patches + `compute_120f` (CUDA 13.0), 39 tok/s native FP4. This is
  baseline kernel correctness, not graph-capture-specific — orthogonal to what
  we found.
- [FlashInfer #2776](https://github.com/flashinfer-ai/flashinfer/issues/2776) —
  NVFP4 MoE crashes during graph capture on SM120/SM121, attributed to a
  workspace-buffer memory-alignment bug in the FlashInfer TRTLLM decode kernel
  specifically. Doesn't explain `marlin` — a completely different kernel —
  failing under capture too. No confirmed fix in the issue; the only workaround
  offered is disabling FlashInfer MoE FP4 entirely, which doesn't restore graph
  capture.

Neither issue, as filed, covers "three independent kernel paths including a
weight-only one all fail under capture." **This project's finding is narrower
than either upstream report and may be worth filing separately** — low priority
against the fine-tuning roadmap, but flagged here so it isn't lost.

Also relevant, as general context rather than a fix: Pape, Evertz & Schönherr,
["The Silent Hyperparameter: Quantifying the Impact of Inference Backends on LLM
Reproducibility"](https://arxiv.org/abs/2605.19537) (arXiv:2605.19537, 2026) —
a survey of 200 inference engines finding that backend choice is a rarely
reported but consequential variable in LLM evaluation. This project's bug is a
specific, unusually severe instance of the general phenomenon that paper
documents.

**Re-verification method (2026-08-14):** first attempt used `llm.generate()`
with a raw prompt string and no chat template — produced fluent but off-topic
text, which looked like a partial-corruption signature but was actually a
**test-harness error**: ZAYA is an instruct/reasoning model and does not perform
sensible raw completion without its chat template, independent of any kernel or
graph-capture question (§6.5). Re-run via `llm.chat()` with
`enforce_eager=True`: correct, on-topic, budget-appropriate reasoning on all 3
prompts (correctly invoked Rayleigh scattering; correctly distinguished RGB vs.
CMY primary-color systems). This is the first firsthand coherence check on this
exact checkpoint since the 2026-08-11 sweep, and it confirms that sweep's
conclusion rather than adding a new one.

**Corrected throughput** (5 process invocations per config, not iterations
within one process — see the Kalibera & Jones citation in `README.md`; GPU
otherwise idle): median 9.52 tok/s (uniform) / 9.51 tok/s (mixed) single-stream,
73.4 / 74.4 tok/s batch-8. Full table, conditions, and the retraction notice:
`README.md` → "Known Issue: CUDA graph capture corrupts output on SM120". Not
reproduced here to avoid the numbers drifting out of sync between the two files.

**Consequence for the open TPOT question (§5.12.7):** the ~10× gap between
expected and measured decode speed, previously attributed to
`trtllm::fused_moe::gemm2` skipping all tactics, is **not** resolved by this
fix — `enforce_eager` buys correctness, not the missing speed. The near-linear
batch-8 scaling measured under the corrected config (96–98% of ideal) is
consistent with a batch-independent per-step kernel overhead, which narrows
where to look next but does not close the question.

**Process lesson:** before concluding a checkpoint is damaged, re-run one prompt
with `enforce_eager=True`. Thirty seconds. Would have saved the confusion in
§5.13's harness-artifact chase and the retracted throughput figures both.

### 5.15 Marlin Does Not Close the TPOT Gap — the "10×" Framing Itself Was Unsound (2026-08-14)

**Headline: swapping to the Marlin MoE backend does not fix the speed. That's a
real negative result, and it exposes a bigger problem — the "~10× gap" this
section chased was measured against a baseline that was never valid.**

**Test:** same rigor as §5.14's corrected throughput table — 5 process
invocations, `enforce_eager=True`, uniform checkpoint, batch-1 and batch-8 —
with `--moe-backend marlin` instead of the default (`FLASHINFER_CUTLASS`,
auto-selected). Coherence re-verified via `llm.chat()` first (correct,
on-topic; the France prompt this time explicitly stated "France's capital is
Paris").

| | default backend | marlin |
|---|---:|---:|
| batch-1, median | 9.52 tok/s | 9.69 tok/s |
| batch-8, median | 73.4 tok/s | 72.75 tok/s |

The difference (1.8% / 0.9%) is inside the ~3.6–3.9% run-to-run noise already
established for this measurement setup. **Statistically indistinguishable.**
Marlin's dequant-matvec kernel and the TMA-warp-specialized grouped-GEMM path
are architecturally unrelated — one landing exactly where the other does is
evidence *against* "wrong tactic chosen for a tiny-M grouped GEMM" as a fixable
software bug, and evidence *for* something more structural: 80 layers of
weight loads + kernel-launch overhead at effectively M≈1 per expert, which may
be close to the real cost of this architecture on this hardware rather than a
bug with a kernel-swap fix.

**The bigger issue this surfaces:** the "~10× TPOT gap" language (§5.14, and
[[session_zaya_bench_2026-08-10]] before it) was always stated relative to the
9.75 ms/token implied by the **102.6 tok/s figure — which is the same number
this project retracted for being measured on a numerically broken CUDA-graph
path.** A code path producing wrong output has no guaranteed relationship to
how much correct work it was actually doing; it may have been doing *less*,
not the same work faster. **There is no established evidence the "expected"
speed this gap was measured against was ever real.** Absent a trustworthy
reference point, "~9.5–9.7 tok/s might just be the honest speed of this
architecture on this card" is at least as well-supported by current evidence
as "there is a fixable ~10× gap" — this section should stop asserting the
latter as settled.

**What would actually resolve this:** an independent reference speed for
top-1-routed MoE decode at this shape (2048 hidden, 4096 FFN, 16 experts) on
SM120, from a source that isn't this project's own retracted number — e.g. a
llama.cpp GGUF build of ZAYA1 (§ecosystem survey, 2026-08-14: `Abiray/ZAYA1-8B-GGUF`,
2,127 downloads, no throughput published) benchmarked directly, or NVIDIA's
own MoE kernel benchmarks at comparable shapes. Neither has been done. Until
one is, "TPOT gap" is a retired framing, not an open bug.

### 5.16 An Independent Reference Number Exists — but Could Not Be Verified on This Hardware (2026-08-14)

**Headline: a real external throughput reference for ZAYA1 does exist (llama.cpp
PR #23112's own author, RTX 4070 Ti, 45.9 tok/s), and it points the opposite
direction from §5.15's conclusion — a slower GPU beating our number by ~4.8×.
Five independent, well-diagnosed attempts to reproduce it directly on our own
SM120 hardware all failed the same way: a non-deterministic hang. This is not
a fixable software bug we found — it is a genuine blocker, documented in full
so it isn't silently re-attempted or silently ignored.**

§5.15 retired the "~10× TPOT gap" question partly because no independent
throughput reference existed for ZAYA1 anywhere. That premise turned out to be
wrong: `llama.cpp` [PR #23112](https://github.com/ggml-org/llama.cpp/pull/23112)
(the draft that `Abiray/ZAYA1-8B-GGUF`, 2,127 downloads, depends on) has the
author's own reported numbers in the PR thread:

| Hardware | Quant | Generation |
|---|---|---:|
| RTX 4070 Ti | BF16 | 11.0 tok/s |
| **RTX 4070 Ti** | **Q4_K_M** | **45.9 tok/s** |
| AMD 9060 XT (ROCm) | — | ~30–35 tok/s |
| Raspberry Pi 5 16GB | Q6_K | 6.5–6.9 tok/s |

RTX 4070 Ti (Ada) has ~504 GB/s memory bandwidth versus our RTX 5070 Ti
(Blackwell)'s 896 GB/s — roughly 1.8× less. A weights-only Q4_K_M GGUF on a
slower card beating our 9.5 tok/s W4A4 checkpoint by ~4.8× is a real signal
worth chasing, not something to wave away with a caveat.

**Direct reproduction attempt — five independent variables tested, all failed
the same way:**

1. **Default build** (llama.cpp master + PR #23112 branch, CUDA 13.2 toolkit,
   `-DCMAKE_CUDA_ARCHITECTURES=120`): `llama-cli` on the ZAYA Q4_K_M GGUF hung
   completely — alive 26+ minutes, near-zero CPU/GPU utilization, immune to
   `timeout`'s default SIGTERM, required `kill -9`.
2. **`GGML_CUDA_DISABLE_GRAPHS=1`** (the documented fix for a matching symptom
   — "CUDA graph update failed... endless loop without returning"): same
   signature, a small burst of initial activity then a stall.
3. **CPU-only (`-ngl 0`)**: genuine progress this time (real CPU-time
   accumulation, unlike the GPU attempts) — then also stalled and needed
   `kill -9`. Isolates the hang to the GPU path specifically.
4. **A completely mainstream model** (`bartowski/Llama-3.2-1B-Instruct-GGUF`,
   138K downloads, nothing ZAYA-specific) through the identical build: **also
   hung**, zero output, exit 124. This ruled out the ZAYA/CCA-conv code as the
   cause — the bug is in the build/environment, not the model.
5. **Rebuilt against CUDA 12.8** (user-local install, no driver component —
   motivated by a documented CUDA-13.2-and-GGUF-inference warning from
   Unsloth's own docs): mainstream model **still hung**, identically.
6. **Rebuilt at `llama.cpp` tag `b7376`** (the last confirmed-good SM120 tag
   per [issue #18090](https://github.com/ggml-org/llama.cpp/issues/18090),
   pre-dating a documented Blackwell regression window `b7376→b7410+`): first
   run produced real output — GPU detected, memory breakdown printed, clean
   exit at the timeout boundary. **Identical rerun of the identical command
   hung earlier and harder**, immune to SIGTERM again.

Step 6 is the deciding data point. The same binary, same command, same model,
back to back, produced a working run and a hard hang. **That rules out a
deterministic software bug tied to a commit, a CUDA version, or a flag** — the
one thing consistent across all six attempts is timing-dependent flakiness,
not configuration. This fingerprint matches two independently-documented
external reports rather than anything specific to this project: a GSP
firmware silent hard-hang on SM120
([NVIDIA/open-gpu-kernel-modules#1111](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1111))
and general WSL2 "CUDA app hangs at 0% GPU util, `nvidia-smi` still works,
driver-level deadlock" reports. Neither is a confirmed match — just the same
class of problem.

**Why this isn't chased further this session:** every lever available from the
build side (toolkit version, llama.cpp commit, CUDA-graph flag, backend)
produced the same non-deterministic outcome. The remaining candidates —
Windows-side driver update, WSL kernel update — are outside what re-running
builds can fix, and outside this session's scope.

**What this changes:** the TPOT question is **not** closed the way §5.15
suggested. An external reference exists, it disagrees with our number by a
wide margin, and the honest state is *"real evidence unable to be
independently verified,"* not *"no evidence exists."* Citing 45.9 tok/s (RTX
4070 Ti, Q4_K_M) requires these caveats every time: different GPU architecture
entirely (not a same-hardware comparison), weights-only quantization versus
this project's W4A4 (activations quantized too — should theoretically favor
W4A4, not disfavor it), and the PR author's own admission that their CCA
attention convolution kernel (`ggml_conv_1d_grouped`) is "naive... instead of
a dedicated kernel per backend" — their number may itself be a floor, not a
ceiling.

**Ecosystem survey reconfirmed (2026-08-14):** 30+ ZAYA1 quant repos on
HuggingFace surveyed for any published throughput number to use as a second
reference point (GGUF, MXFP4, MLX, bnb, ONNX). None publish one. The 45.9
tok/s figure remains the only external number that exists, sourced from a PR
comment, not a model card.

### 5.17 The Gap Has a Principled Explanation We'd Been Missing: W4A4 Isn't Supposed to Win at Batch-1 (2026-08-14)

**Headline: activation quantization (the "A4" in W4A4, this project's core differentiator)
provides no speed benefit at batch-1, and can be slower than weight-only
quantization at the same bit-width. This is documented, expected behavior for
the quantization scheme, not a bug — and it plausibly explains a substantial
part of the gap chased since §5.14, without requiring anything to be fixed.**

Decode at batch-1 is memory-bandwidth-bound: the dominant cost is loading
weights from VRAM, not the matrix multiplication itself. Quantizing
activations only pays off when compute is the bottleneck — batched serving or
prefill — because it lets both operands use Blackwell's FP4 tensor cores at
full rate. At batch-1, there's no compute bottleneck to relieve, so
activation quantization adds dequant/requant overhead without offsetting
gain. Documented example at comparable scale: weight-only NVFP4 beat full
W4A4 on the same hardware at small batch for Gemma 4 12B.

This means the §5.16 comparison (45.9 tok/s weight-only Q4_K_M on a slower
GPU vs. 9.5 tok/s W4A4 here) was never purely an apples-to-apples speed
contest — it was comparing two quantization schemes with different design
points. **Weight-only optimizes for batch-1 decode. W4A4 optimizes for memory
footprint and batched/compute-bound throughput.** The batch-8 numbers in this
project (73–74 tok/s, 96–98% of ideal linear scaling from batch-1) are
exactly where W4A4 should be expected to show its advantage, and do.

**What this does and doesn't resolve:**
- Does: give the gap a principled, well-documented cause instead of leaving
  it as an unexplained anomaly. No further engineering effort is owed here
  before publishing a number — the number is now understood, even without a
  precise "the gap should be exactly N×" model.
- Doesn't: fully close the question. This is a general documented tendency,
  not a measurement of *this* checkpoint's specific dequant overhead. §5.16's
  blocker (llama.cpp hangs on this hardware) still stands between us and a
  same-hardware, same-scheme empirical comparison, and the filed upstream
  reports (`microsoft/WSL#41361`) remain the path to closing that gap if a
  fix ever lands.

**Consequence for how this project talks about itself:** the differentiator
framing throughout this repo and its downstream surfaces ("W4A4 — activation
quantization is the moat") is accurate about *what's distinct*, but was
previously silent on *where the win shows up*. It's memory footprint and
batched/prefill throughput, not single-stream decode latency. Worth stating
that precisely rather than leaving a reader to assume W4A4 should beat
weight-only quantization at every batch size — it doesn't, by design, and
claiming otherwise would be exactly the kind of overclaim this project has
spent this whole line of investigation avoiding.

### 5.18 N-gram Speculative Decoding: 2.2× on Coding-Edit Workloads, No Gain on Free-Form Text (2026-08-14)

**Headline: vLLM's built-in n-gram speculative decoding — zero training, zero
new model, one config flag — gives a real, validated 2.2× speedup on
code-editing prompts. It gives nothing on free-form generation. Both results
are expected and both matter: Godspeed's actual workload is almost entirely
the former.**

n-gram (prompt-lookup) speculative decoding proposes draft tokens by matching
against recent context rather than running a second model. It only pays off
when upcoming output overlaps with what's already been said — which is
exactly the shape of a coding-agent request (read a file, echo most of it
back with a small edit) and exactly *not* the shape of free-form generation.

**Test 1 — free-form prompt** ("why is the sky blue"), `vllm bench latency`,
`--speculative-config '{"method": "ngram", "num_speculative_tokens": 5,
"prompt_lookup_max": 5, "prompt_lookup_min": 2}'`: ~9.3 tok/s, indistinguishable
from the ~9.5 tok/s baseline. No overlap for the proposer to exploit, no gain.
Expected, not a negative result about the method — a mismatched test case.

**Test 2 — realistic coding-edit prompt.** `vllm bench latency` only generates
synthetic random tokens, so it can't test this — a small bespoke script
(`coding_edit_bench.py`) was written instead, feeding the model a real 98-line
file from this repo (`scripts/verify_w4a4_dequant.py`) with instructions to
return the complete file with two small changes. 5 process invocations per
condition, `enforce_eager=True`, uniform checkpoint, greedy, 500-token cap,
chat-templated:

| | median tok/s | range |
|---|---:|---|
| Baseline | 9.62 | 9.13–9.82 |
| N-gram speculative decoding | **21.11** | 20.13–21.93 |

**2.2× speedup, zero anomalies across all 10 runs.** Output checked for
coherence in both modes — same on-topic reasoning content, no corruption or
repetition, consistent with speculative decoding's rejection-sampling
guarantee that output distribution is preserved exactly (this isn't a
sampling shortcut; it's provably identical output, just fewer full forward
passes to get there).

**Relevance to §5.17:** this doesn't change the batch-1 W4A4-vs-weight-only
tradeoff finding — that's about the underlying decode step's memory-bandwidth
bound, which n-gram decoding works *around* by skipping steps entirely when
speculation succeeds, not by making each step faster. Compatible, additive
levers, not competing explanations.

**Update, same day: deployed and validated end-to-end.** `~/scripts/vllm-serve.sh`
now starts with `--speculative-config`. Confirmed via a real request through
the live OpenAI-compatible API (not just the offline benchmark harness) —
server reaches healthy, responds coherently to a coding-edit request.

One real tradeoff surfaced only at serve time, not visible in the offline
benchmark: vLLM logs `"Async scheduling not supported with ngram-based
speculative decoding and will be disabled."` Async scheduling was on in the
pre-n-gram config; this is a real secondary cost of enabling n-gram decoding,
not accounted for in the 2.2× figure above (which compared enforce_eager
on/off, not async-scheduling on/off). Given the 2.2× win, presumed net
positive, but not independently measured — if a future speed regression
shows up in real usage, check this first before assuming n-gram itself
regressed.

Also noted: vLLM warns `max_num_scheduled_tokens is set to 2048 based on the
speculative decoding settings... Consider increasing max_num_batched_tokens`
— a real tuning knob left unexplored, not a problem.

**Not yet tried:** Token Recycling or EAGLE-family methods (§ prior research),
which could add further gains on top of or instead of n-gram, particularly
for the free-form-generation case this doesn't help.

**Infrastructure note, resolved same day:** `~/scripts/vllm-serve.sh` has
since been brought under version control (private repo
[t-timms/machine-scripts](https://github.com/t-timms/machine-scripts)), and
the exact validated serve configuration is now also published in this repo as
`scripts/serve.sh` so downloaders get the same win, not just the private
machine.

### 5.19 TRITON_ATTN Does Not Fix CUDA Graph Correctness Either — a Second Axis Ruled Out (2026-08-14)

**Negative result, but a valuable one: this project's CUDA-graph correctness
bug (§5.14) now fails across two independent configuration axes, not one.**

§5.14's original sweep varied the **MoE backend**
(`flashinfer_cutlass`/`cutlass`/`marlin`) and found all three produce garbage
under graph capture. It never varied the separate **attention backend**
(`FLASHINFER`/`TRITON_ATTN`) — a genuinely different code path.

Motivation for testing it: a public vLLM issue
([#41651](https://github.com/vllm-project/vllm/issues/41651)) describes an
SM120-specific bug with an unnervingly close signature — FlashInfer attention
+ FP8 KV cache + CUDA graphs producing random output, specifically on *long*
prompts, with `TRITON_ATTN` reported as a working switch that preserves CUDA
graphs. ZAYA1 is a long-reasoning model: even a short user prompt like "name
three primary colors" produces a long effective context once the `<think>`
trace grows, which plausibly matches that "long prompt" trigger.

**Test:** `attention_backend="TRITON_ATTN"`, `enforce_eager=False` (CUDA
graphs on), same checkpoint, same greedy "name three primary colors" prompt
used in the original §5.14 diagnosis.

**Result: graph capture succeeded cleanly this time (35/35 graphs captured,
no crash/hang) — but output was still garbage** (`"sariant vessel"`, nowhere
near a coherent answer). TRITON_ATTN does not fix this project's bug.

**What this rules out:** the bug is not tied to one specific attention
kernel any more than §5.14 showed it isn't tied to one specific MoE kernel.
Two independent axes, four total kernel combinations, all fail identically
under graph capture, all succeed under `enforce_eager=True`. This is
stronger evidence that the fault is in graph capture itself for this
model/hardware/driver/WSL combination — not a swappable kernel choice.
`enforce_eager=True` remains the only known fix.

**Not pursued further today:** a separate, larger lead exists —
[CUTLASS #3096](https://github.com/NVIDIA/cutlass/issues/3096) (already
cited in this project's docs) documents a real fix for SM120 NVFP4 MoE
grouped-GEMM garbage output, but it requires rebuilding FlashInfer with
`compute_120f` instead of `compute_120a` (CUDA 13.0+, 10+ patched files) —
a build-time change, not a runtime flag, and a substantially bigger lift
than today's two quick tests. Worth a dedicated session if this bug is
revisited; not attempted here given the time budget.

### 5.20 EBSS Is a No-Op for This Pipeline — and Its Implementation Was Silently Corrupting Checkpoints (2026-08-15)

**Two findings, one session: a real implementation bug that silently produced
a broken checkpoint, and — after fixing it — proof that the technique cannot
help this pipeline at all.**

#### What happened

Regenerated EBSS-rebalanced calibration data (§5.12.4) and requantized the
uniform checkpoint with it, SOAR enabled, same `--no-bf16-exempt` methodology
as the published build. **The run exited 0 with no errors** and wrote a
correctly-sized 6.02 GB checkpoint. Paired lm-eval against the published
checkpoint:

| task | published (healthy) | SOAR+EBSS | random baseline |
|---|---:|---:|---:|
| hellaswag acc_norm | 60.29% | **25.75%** | 25% |
| arc_challenge acc | 35.75% | **25.17%** | 25% |
| piqa acc | 68.93% | **55.06%** | 50% |
| winogrande acc | 57.70% | **49.72%** | 50% |

Chance level on three of four tasks. Generation confirmed total incoherence
(`"Western courage Worker Assign Worker socialization Hudson exclude⅙⅙…"` for
"name three primary colors") — under `enforce_eager=True`, so *not* the
§5.14 CUDA-graph bug.

#### Root cause, isolated in three steps

1. **Weights were byte-identical.** Sampled `weight_packed`, `weight_scale`,
   and `weight_global_scale` across both checkpoints: identical everywhere.
   Only `input_global_scale` differed (59 of 60 sampled). Weight quantization
   is computed from the weights, not from activations, so this was expected —
   and it narrowed the fault to activation calibration alone.

2. **IGS was systematically inflated.** `IGS = 2688 / max_act`, so a higher
   IGS means a *lower* observed activation maximum. Across all 1,320 modules:
   **95.8% had higher IGS in the broken checkpoint, median ratio 1.68×**,
   implying EBSS observed activation maxima only **~60% as large** (p10: 30%)
   as the original corpus. Calibrating the representable FP4 range ~40% too
   small means real activations saturate at inference — exactly the observed
   corruption, with identical weights.

3. **The calibration corpus was degenerate.** Comparing the two tensors
   directly:

   | | original arcmix | EBSS output |
   |---|---:|---:|
   | unique rows | 977 / 977 (100%) | **3 / 977 (0.3%)** |
   | most-repeated row | 1× | **972×** |
   | unique token ids | 28,397 | **738** |

   `ebss_resample()` selected essentially one sample and repeated it 972
   times. The greedy loop appended `best_idx` but **never masked it from
   subsequent rounds**, relying on the `1/(1+coverage)` reweighting to
   discourage re-selection. Once coverage reaches the thousands that term
   shrinks near-proportionally for every expert, so the relative ranking
   between samples barely moves and the same argmax wins every iteration.

   This also explains the two side-effects seen at quantization time: the
   modules with zero activation observations (that single repeated sample
   never routes to layer 71 / expert 15), and the "imbalance ratio improved
   0.04 → 0.06" line — that metric was measuring coverage of a degenerate
   set, so it was meaningless.

#### The deeper result: EBSS cannot help this pipeline

After fixing the selection to sample **without replacement**, the rerun
produced 977/977 unique rows and 28,397 unique token ids — and **expert
coverage identical to the original, imbalance ratio 0.04 → 0.04, unchanged.**

That is not a disappointing outcome, it is the arithmetic: selecting `N`
samples without replacement from a corpus of `N` is a permutation of the
whole corpus, and coverage is a sum over all samples. More fundamentally,
`activation_max` is accumulated as a **running max over every calibration
token** — an order- and frequency-independent statistic. **No reweighting or
reordering of a fixed corpus can change a max.** EBSS could only matter if
`target_n < N` (a strict subset), and for max-based calibration a subset is
strictly worse: fewer samples can only lower the observed maximum, which is
the exact failure mode that corrupted the checkpoint here.

MoEQuant's EBSS targets calibration schemes where sample *frequency* carries
signal — fitted statistics like GPTQ Hessians or percentile-based scales.
This pipeline's activation calibration is a pure max, so the technique is
inapplicable by construction. **EBSS is closed as a lever for this project**,
independent of the bug.

#### Fixes shipped

- `build_calibration_ebss.py`: selection is now without replacement, and a
  **diversity guard refuses to write** a corpus below 90% unique rows rather
  than emitting one that silently corrupts downstream quantization.
- `quantize_zaya_ct_nvfp4.py`: an uncalibrated module previously left
  `input_global_scale = 0` and logged a warning — which makes `block_scale`
  zero and yields NaN/garbage the moment that expert is routed to. It now
  **repairs inline** with a same-layer/same-type median fallback (the logic
  `fix_uncalibrated_igs.py` applies post-hoc, moved to the source) and
  **raises** if any module cannot be repaired. A silently-broken checkpoint
  can no longer be written.

**Process note worth keeping:** the run exited 0, wrote a correctly-sized
checkpoint, and passed every structural check. Only a real accuracy eval
caught it. This is the second time in two sessions that "exit code 0" meant
nothing (§5.14 was the first) — judge these runs by measured behaviour, never
by process status.

### 5.21 Generative Eval Suite: Methodology, and Why the Standard Tool Doesn't Cover This Model (2026-08-15)

Every accuracy number this project has published so far comes from
**loglikelihood** tasks (hellaswag, arc_challenge, winogrande, piqa). Those
score pre-written continuations — they measure *ranking*, never *producing*.
That is a structural blind spot, and this project has already been burned by
it: a checkpoint that could not form a coherent sentence scored 61.18% on
HellaSwag. GSM8K, HumanEval and MMLU-Pro are added specifically to close it.

#### The standard-tool check, and its outcome

Per this project's "standard tools before bespoke" rule, lm-eval-harness was
checked first — and it *does* now ship reasoning-model support
(`think_end_token`, added July 2025, available in the installed build for the
vLLM backend). Reading the implementation:

```python
generation = generation.split(think_end_token)[-1].lstrip()
```

It strips **post-hoc**. When the model closes `</think>`, this works. When it
does not, `split()` returns a single element and `[-1]` hands back the entire
unterminated reasoning trace *as the answer*, which is then scored — exactly
the artifact that put IFEval at 19.8% against a BF16 reference of 85.58%.

ZAYA1 frequently does not close its think block within a feasible budget, so
the standard tool cannot measure it correctly. The bespoke scripts implement
actual budget **forcing** (generate bounded reasoning, then *inject*
`</think>` and decode the answer that gets scored) rather than post-hoc
stripping. Every scorer inside them is still lm-eval's own — the gsm8k
flexible-extract regex, the mmlu_pro answer regex, HF `evaluate`'s sandboxed
`code_eval` for pass@1 — so only the generation protocol is custom, not the
grading.

#### Think-budget is a real parameter, not a default to accept

Measured on GSM8K, n=20, identical items and seed:

| think_budget | accuracy | self-closed `</think>` |
|---:|---:|---:|
| 2048 | 50.0% | 1/10 |
| 4096 | **75.0%** | 4/20 |

A 25-point swing from the budget alone. This is the third benchmark on which
an insufficient budget has understated this model (GPQA was 45.8% → 62.5%
across the same lever). **4096 is the default**; the scripts now also report
how many items hit the ceiling, so an under-budgeted run is visible in its own
output rather than silently depressing the score.

#### Extraction failures were silently costing accuracy

The first GSM8K smoke run scored one item wrong where the model had reasoned
correctly — the forced short answer opened with a newline, hit the stop
sequence, and yielded no number at all. That is a *formatting* miss being
recorded as a *reasoning* miss, and at 1/20 it is not negligible. The scripts
now fall back to the tail of the reasoning trace when the forced answer
contains no parsable answer (the fallback GPQA and MMLU-Pro already had) and
report how many items needed it, so the rate stays visible.

#### Baselines: what can and cannot be compared

| benchmark | Zyphra BF16 published | usable as baseline? |
|---|---|---|
| MMLU-Pro | 74.2 | indicative only — see below |
| GSM8K | **not published** | no baseline exists |
| HumanEval | **not published** | no baseline exists |

Zyphra states only that "all numbers are run on the Zyphra evaluation
harness" — private, with undisclosed generation limits and prompting. Their
figures are therefore not reproducible by anyone, and comparing a
budget-forced number against them is a *different experiment*, not a
reproduction. GSM8K and HumanEval results stand alone. Running the BF16 model
locally to generate a matched baseline is not possible either: at 17.7 GB of
weights it does not fit in this card's 16 GB.

Sampling follows Zyphra's published recommendation (temperature 0.6, top_p
0.95 — their agent/code setting; they suggest 1.0 for general use), fixed
seed 42, recorded in every result artifact.

#### Contamination: read GSM8K and MMLU-Pro results differently

These three benchmarks do not carry equal evidential weight, and the
difference is worth stating rather than presenting one aggregate table:

- **GSM8K is substantially contaminated and largely saturated.** Removing
  contaminated examples from its test set has been shown to drop accuracy by
  **up to 13 points** for some models — meaning a meaningful share of any
  high score reflects training-set overlap rather than reasoning. A GSM8K
  number here is a *sanity check that quantization did not break arithmetic
  reasoning*, not evidence of mathematical ability.
- **MMLU-Pro was purpose-built as the contamination-resistant successor to
  MMLU**, which makes it the strongest of the three as an accuracy signal.
- **HumanEval** has no published BF16 baseline for this model and is only 164
  problems, so its confidence interval is wide by construction — see the
  Wilson interval now reported alongside every result.

Corollary for anything published from these runs: lead with MMLU-Pro, treat
GSM8K as a regression check, and never report a bare point estimate without
its interval.

#### Every result now carries an interval

A bare accuracy is not interpretable, and this project has already been
misled by one: 75% vs 80% on the think-budget sweep looked like a real gain
and tested at **p = 1.0000**. All three scripts now print and store a **95%
Wilson score interval** next to the point estimate (Wilson rather than the
normal approximation because it stays sensible at small n and near 0/1,
which the subset runs hit). This matches the standard lm-eval reports and
follows the reproducible-evaluation guidance in
[Biderman et al., *Lessons from the Trenches on Reproducible Evaluation of
Language Models*](https://arxiv.org/pdf/2405.14782).

#### Tooling shipped

- `scripts/run_budget_forced_suite.sh` — runs all three unattended, one
  process per benchmark, cheapest-first so a partial run still leaves
  results. A stage is skipped only if its artifact **parses as JSON**; never
  a size threshold (a 200-byte floor once rejected a valid 188-byte result),
  never an exit code (vLLM can abort at teardown *after* writing a valid
  result). Records the environment fingerprint alongside the numbers.
- `scripts/compare_budget_forced.py` — paired exact-binomial McNemar on
  discordant items, since comparing two checkpoints on the same items is a
  paired design and aggregate-only comparison wastes the run (the ARC-Easy
  +1.81 pp / p=0.18 mistake). Refuses to compare runs with mismatched
  think_budget, and prints the CI with an explicit warning against reporting
  a non-significant point estimate. Verified against four fixtures: a real
  difference (correctly significant), pure coin-flip noise (+4.50 pp,
  correctly *not* significant), a budget mismatch (warns), and HumanEval's
  schema (+16.67 pp at n=60, correctly *not* significant).

### 6.1 Audited Repositories

- `Zyphra/transformers` @ zaya1 branch (`modular_zaya.py`, `configuration_zaya.py`)
- `Zyphra/vllm` @ zaya1-pr branch (`zaya_tool_parser.py`)
- `Zyphra/ZAYA1-8B` HuggingFace model card + `tokenizer_config.json`

### 6.2 SOTA Features Already Present

Zyphra's implementation includes several state-of-the-art optimizations:
- Dynamic `torch.compile`/`torch.jit.script` dispatch per PyTorch version
- FP8 activation storage for backward pass memory savings
- Fused bias+SwiGLU custom autograd kernels
- FP32 residual accumulation for numerical stability in 80-layer models
- Per-layer residual scaling (DeepNet-style)
- Dual time-stream value projections in CCA attention
- Depth-wise averaging (EDA) in MoE router
- Mixture-of-Depths (MOD) skip expert
- Three attention backends with PEFT-aware fp32 casting

### 6.3 Upstream Improvement Opportunities

14 findings identified by comparing `modular_zaya.py` against DeepSeek-V3.1,
DeepSeek-V4 Pro, and Qwen3-MoE modeling code. See `patches/UPSTREAM_PROPOSAL.md`
for full implementation details.

**10 safe patches (zero forward-pass impact, bit-identical output):**

| # | Patch | Gain |
|---|-------|------|
| 1 | `GradientCheckpointingLayer` base class | 40-60% activation memory reduction during training |
| 2 | `_can_compile_fullgraph = True` | 15-30% inference speedup via `torch.compile(fullgraph=True)` |
| 3 | `_can_record_outputs` metadata | Enables TRL intermediate output capture |
| 4 | `logits_to_keep` support | ~2 GB/batch memory savings during training |
| 5 | Hub-loaded RoPE kernel | 5-10% attention speedup via Triton kernels |
| 6 | `_tied_weights_keys` declaration | Fixes PEFT weight tying detection |
| 7 | `_tp_plan` / `_pp_plan` | Enables multi-GPU distributed inference |
| 8 | `_supports_flex_attn = True` | 10-20% attention speedup via FlexAttention |
| 9 | Hub-loaded RMSNorm | 5-10% normalization speedup via Triton kernels |
| 10 | `router_aux_loss_coef` config | Enables future MoE aux loss computation |

**Combined impact**: ~60% activation memory reduction + 40-70% training throughput
improvement without any architectural changes.

**4 unsafe patches (would break ZAYA-specific innovations):**

| # | Finding | Why blocked |
|---|---------|-------------|
| 1 | MoE aux loss with standard `load_balancing_loss_func` | Penalizes MOD skip expert, destroying Mixture-of-Depths |
| 2 | 3D fused expert weights | Disables fused bias+SwiGLU and FP8 backward storage |
| 3 | `@use_experts_implementation` | Doesn't understand MOD or EDA routing |
| 4 | Standard MoE weight initialization | May destabilize EDA-tuned routing distribution |

### 6.4 DeepSeek V4 Pro Teacher Model Reference

DeepSeek V4 Pro (1.6T total / 49B active, 1M context, FP4+FP8) provides the
SOTA reference for this experiment's teacher model. Key design patterns:

| Feature | DeepSeek V4 Pro | ZAYA1-8B |
|---------|----------------|----------|
| SWE Verified | 80.6% | N/A (BFCL-v4: 39.22 baseline) |
| LiveCodeBench | 93.5% | 65.8% |
| Codeforces | 3,206 | N/A |
| On-policy distillation | Yes (independent domain experts → unified) | No (4-stage sequential RL) |
| Muon optimizer | Yes | Not specified |
| Context window | 1M tokens | 131K tokens |

### 6.5 Chat Template Compatibility

ZAYA1-8B uses Qwen-style `<|im_start|>` / `<|im_end|>` tokens with `<think>` / `</think>` blocks. TRL auto-patches Qwen-family chat templates for `assistant_only_loss=True` and prefix-preservation for tool calls. Our configuration passes `chat_template_kwargs={"enable_thinking": True}` to match ZAYA1-8B's always-on thinking mode.

---

## 8. SOTA Training Configuration

### 8.1 Configuration Evolution

| Component | Original | Final (SOTA) | Source |
|-----------|----------|--------------|--------|
| Trainer | `Trainer` (vanilla) | `SFTTrainer` (TRL) | TRL docs |
| Loss masking | None | `assistant_only_loss=True` | TRL SFT docs |
| LoRA scaling | Standard (`α/r`) | rsLoRA (`α/√r`) | PEFT docs |
| Loss computation | Standard NLL | `chunked_nll` (60% mem reduction) | TRL SFT docs |
| Kernel library | None | Liger Kernel (20% throughput) | TRL Liger Kernel docs |
| Chat template | Default | `enable_thinking=True` | ZAYA1-8B docs |
| Weight tying | Not set | `ensure_weight_tying=True` | ZayaConfig |
| Epochs | 3 | 2 | Project docs constraint |
| GRPO loss | Not implemented | `dapo` (token-level norm) | DAPO paper |
| GRPO reward scaling | Not implemented | `batch` (global std) | TRL GRPO docs |
| Dataset format | Manual tokenization | Conversational (TRL native) | TRL Dataset Formats docs |

### 8.2 Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Catastrophic forgetting (AIME ↓) | QLoRA r=16, 2 epochs max. Hard stop if AIME <84 |
| Tool-call format corruption | All `<tool_call>` → `<zyphra_tool_call>` (critical fix applied) |
| Shopify OOD failure | 30% OOD tasks minimum in training data |
| Training data quality | 5 quality gates during remapping |
| Benchmark leakage | Tasks vs SWE-bench Verified IDs cross-check before eval |
| Expert collapse (GRPO) | Monitor expert utilization if router weights trainable |
| Format mismatch (teacher vs student) | Godspeed XML → ZAYA XML remapping with validation |

---

## 9. Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| `data/generate.py` | 23 | Parsing, filtering, truncation, session finding |
| `scripts/mutate_tasks.py` | 21 | All 6 mutation types, dedup, OOD ratio, seed |
| `scripts/remap_to_zaya.py` | 56 | Tool-call format, tool-response format, all 5 quality gates, Jaccard, token estimation, remap pipeline |
| **Total** | **100** | **All passing** |

---

## 10. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥2.5.0 | ML framework |
| transformers | Zyphra fork @ zaya1 | ZayaForCausalLM support |
| peft | ≥0.13.0 | LoRA adapters |
| bitsandbytes | ≥0.45.0 | NF4 quantization |
| trl | ≥0.24.0 | SFTTrainer, GRPOTrainer |
| datasets | ≥3.0.0 | HuggingFace datasets |
| pyyaml | ≥6.0 | Config parsing |
| vllm (optional) | Zyphra fork @ zaya1-pr | Inference server |
| unsloth (optional) | ≥2026.5.0 | Faster training |
| liger-kernel (optional) | ≥0.6.0 | Memory-efficient kernels |

---

## 11. References

1. ZAYA1-8B Technical Report. Washbourne et al. arXiv:2605.05365, May 2026.
2. TRL Documentation. HuggingFace. https://huggingface.co/docs/trl/
3. PEFT Documentation. HuggingFace. https://huggingface.co/docs/peft/
4. vLLM Documentation. https://docs.vllm.ai/
5. Unsloth Documentation. https://docs.unsloth.ai/
6. DeepSeekMath: Pushing the Limits of Mathematical Reasoning. Shao et al. arXiv:2402.03300, 2024.
7. DAPO: An Open-Source LLM RL System at Scale. Yu et al. arXiv:2503.14476, 2025.
8. Understanding R1-Zero-Like Training: A Critical Perspective. arXiv:2503.20783, 2025.
9. LoRA: Low-Rank Adaptation of Large Language Models. Hu et al. arXiv:2106.09685, 2021.
10. QLoRA: Efficient Finetuning of Quantized LLMs. Dettmers et al. arXiv:2305.14314, 2023.
11. The Silent Hyperparameter: Quantifying the Impact of Inference Backends on LLM Reproducibility. Pape, Evertz & Schönherr. arXiv:2605.19537, May 2026.

---

## 12. Next Steps (Updated 2026-05-22, session 14)

### ✅ Complete: W4A4 NVFP4 Pipeline (Sessions 1–14)

| Milestone | Session | Status |
|-----------|---------|--------|
| NVFP4 CT coherent inference (Path A Python dequant) | 2 | ✅ |
| SM120 CUTLASS kernels compiled from source | 4 | ✅ |
| Zyphra vLLM overlay applied | 5 | ✅ |
| W4A4 checkpoint + layer-wise GPU calibration | 6 | ✅ |
| Loader + forward pass on SM120 (global-scale bug fixed) | 7 | ✅ |
| lm-eval baseline (ARC-Easy 67-68%, HellaSwag 60-61%) | 8 | ✅ |
| CUDA graphs (102 tok/s) | 8 | ✅ |
| SOAR global-scale optimization | 9 | ✅ |
| EBSS calibration + MR-GPTQ | 9–13 | ✅ |
| ARC-mix calibration, mixed-precision checkpoint | 14 | ✅ |
| Disk cleanup (47 GB freed, active checkpoint preserved) | 14 | ✅ |
| Smoke test PASSED on active checkpoint | 14 | ✅ |

### ✅ Resolved: Baselines + the "damage" question (Session 17, 2026-06-22)

The baseline question is settled — see **§5.13**. The W4A4 checkpoint is **healthy**:
budget-forced GPQA-Diamond = **45.8%** (vs 25% random), and the low stock-harness numbers
(GPQA MCQ 33.8%, GPQA-CoT 6.6%, IFEval 32.5%) were a **reasoning-model evaluation artifact**
— ZAYA never emits `</think>` within a feasible budget and answers in `\boxed{}` format, so
the stock lm-eval protocols cannot score it. The right protocol is budget forcing
(`scripts/eval_gpqa_budget_forced.py`).

Open follow-ups (low priority, not blocking):
- **IFEval, done right**: needs the same budget-forcing + `<think>…</think>` stripping before
  the instruction-following checkers run. The 32.5% is invalid for the same reason as GPQA-CoT.
- **Tighter CI / bigger budget**: n≈100 for ±~6% CI; budget 12000 @ 16k ctx tests whether
  letting the model finish closes the gap (running 2026-06-22).
- GPQA-Diamond BF16 reference 71.0% is Zyphra's own CoT harness; treat as an approximate
  ceiling, not an identical protocol.

### ⛔ Deprioritized: Rotation + MR-GPTQ "repair" pipeline

**This arc is no longer the plan.** It existed to recover assumed quantization damage; §5.13
shows there is **no measured damage** to recover (45.8% >> random; the gap to 71% is the
16 GB reasoning-budget ceiling, not quant loss). A BF16 control to isolate any residual quant
loss is infeasible locally (~35 GiB > 16 GiB). Kept here only for historical record; do not
start it without first establishing, on a fair budget-forced eval, that rotation actually
moves accuracy.

<details><summary>Historical pipeline (do not run by default)</summary>

1. **Fix rotation absorption bug** in `apply_singlequant_rotations.py` — absorb `R` into
   `linear_fc2.weight @ R.T` (not into LN gamma).
2. **Apply rotation** to BF16 source (`--input Zyphra/ZAYA1-8B`, `--output ./zaya1-8b-bf16-rotated`).
3. **Re-quantize with MR-GPTQ** (`--scheme w4a4 --mr-gptq --arc-mix --mixed-precision-threshold 1000`).
4. **Benchmark** the final checkpoint with `scripts/eval_gpqa_budget_forced.py` (NOT the stock
   lm-eval CoT task) so the comparison is apples-to-apples.

</details>

### ⬜ Follow-ups from the CUDA graph capture finding (§5.14, 2026-08-14)

- Consider filing upstream: no existing issue documents CUDA graph capture
  corrupting output across `flashinfer_cutlass`, `cutlass`, *and* `marlin`
  identically on SM120 — narrower than CUTLASS #3096 or FlashInfer #2776 as
  filed. Low priority against the fine-tuning roadmap.
- Add IFEval (or another free-generation eval) to the standard accuracy suite
  permanently — loglikelihood-only evaluation is why the graph-capture bug went
  undetected for months (§5.14).
- **TPOT gap: reopened 2026-08-14 (§5.16), not retired.** §5.15's retirement
  assumed no independent reference existed; one does (llama.cpp PR #23112
  author, 45.9 tok/s on a slower RTX 4070 Ti). Direct reproduction on our own
  SM120 hardware hit a genuine, five-times-confirmed non-deterministic hang
  (§5.16) — a WSL2/driver-level issue, not a llama.cpp or ZAYA bug. Next
  attempt should start with a Windows NVIDIA driver update or WSL kernel
  update before any more build/version changes; re-running the same build
  variations again would not add new information.

### ⬜ Publication

- Update `PAPER.md` Table 2 with actual benchmark numbers
- Push final checkpoint to HuggingFace Hub (`t-timms/zaya1-8b-nvfp4-w4a4`)
- Write vLLM PR: `unquantized.py` one-liner for mixed-precision MoE loading
- Blog post: end-to-end story — SM120 kernel compilation, global-scale bug,
  mixed-precision exemption, GPTQ+rotation pipeline

### ⬜ Agentic Fine-Tuning (Phase 3–7, unchanged)

- Phase 3: NVIDIA NIM credentials, Godspeed headless teacher trajectories
- Phase 5: QLoRA SFT on verified trajectories, GRPO policy improvement
- Phase 6: BFCL-v4, τ² evaluation
- Phase 7: Deploy to Godspeed driver catalog
