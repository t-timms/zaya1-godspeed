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

- [Zyphra/ZAYA1-8B on HuggingFace](https://huggingface.co/Zyphra/ZAYA1-8B)
- [Technical Report (arXiv 2605.05365)](https://arxiv.org/abs/2605.05365)
- [Zyphra Blog Post](https://www.zyphra.com/post/zaya1-8b)
- [Zyphra vLLM Fork](https://github.com/Zyphra/vllm/tree/zaya1-pr)
- [Zyphra Transformers Fork](https://github.com/Zyphra/transformers/tree/zaya1)
- [NVIDIA NIM — DeepSeek V4 Pro](https://build.nvidia.com/deepseek-ai/deepseek-v4-pro)
- [Godspeed Coding Agent](https://github.com/omnipotence-eth/godspeed-coding-agent)
