# AGENTS.md — zaya1-godspeed

## Project Overview

Fine-tuning Zyphra's ZAYA1-8B (760M active / 8.4B total MoE) for agentic multi-turn
tool calling using teacher-distilled SFT+GRPO. Teacher: DeepSeek V4 Pro via NVIDIA NIM.
See `RESEARCH.md` for the full research document.

## Key Rules

### Tool-call format (CRITICAL)
ZAYA1-8B uses `<zyphra_tool_call>` / `<zyphra_tool_response>` tags (tokenizer IDs 101-104),
NOT `<tool_call>`. If you see `<tool_call>` anywhere, it's a bug. This was verified against
the official `Zyphra/ZAYA1-8B` tokenizer_config.json and `Zyphra/vllm` zaya_tool_parser.py.

### Zyphra fork transforms
This project depends on the Zyphra transformers fork (@ zaya1 branch) and vLLM fork
(@ zaya1-pr branch). The patches in `patches/` monkey-patch missing ecosystem features.
PR submitted: https://github.com/Zyphra/transformers/pull/2

**Zyphra vLLM fork build**: Previous 6 build attempts silently failed because `nvcc` was not in PATH.
Fix: `export PATH=/usr/local/cuda/bin:$PATH` before running pip install.
The fork builds in ~3 min with CUDA 13.2.

**Python file overlay (alternative to full rebuild)**: The Zyphra fork's Python files
can be copied over stock vLLM 0.20.2 without rebuilding CUDA kernels:
- `vllm/model_executor/layers/mamba/cca.py`
- `vllm/v1/attention/backends/cca_attn.py`
- `vllm/model_executor/models/zaya.py`
- `vllm/tool_parsers/zaya_tool_parser.py`
- `vllm/transformers_utils/configs/zaya.py`

Three additional patches required for stock vLLM 0.20.2 Zaya support:
1. `ModelRegistry.register_model("ZayaForCausalLM", ...)` — zaya.py exists but isn't registered
2. `MambaStateShapeCalculator.cca_state_shape` + `MambaStateDtypeCalculator.cca_state_dtype` — missing methods

### PEFT targets
Only target attention projections: `o_proj`, `linear_q`, `linear_k`, `val_proj1`, `val_proj2`.
Never target expert weights (SequentialMLP). This preserves ZAYA's MoE routing, EDA,
and MOD skip expert. Use `ensure_weight_tying=True` because `tie_word_embeddings=True`.

### Training constraints
- Epochs: 1-2 max. Hard stop if AIME degrades >5% from 89.1%.
- VRAM budget: ~9.5-11.5 GB (QLoRA on 16 GB). NF4 base model uses ~7.2 GB + double quantization saves ~0.36 GB.
- Chat template: `chat_template_kwargs={"enable_thinking": True}` required.
- GRPO uses `loss_type="dapo"`, `scale_rewards="batch"`, `num_generations=4`.
- QLoRA quant: NF4 with double quantization (`bnb_4bit_use_double_quant: true`). Bitsandbytes does not officially support 4-bit on MoE `nn.Parameter` tensors — LoRA on attention projections only (not experts) avoids this.

### Known limitations
- QLoRA + MoE: bitsandbytes 4-bit quantization is not validated for MoE architectures. Our LoRA targets attention projections only, not expert weights, so training is safe but monitor for quality issues.
- NF4 dequant on CCA attention: documented as broken in `COMPATIBILITY.md` — inference via NF4+transformers produces garbage output. Use vLLM for inference.
- Unsloth MoE Triton kernels: would provide 2-12x faster training but ZAYA's `SequentialMLP` architecture isn't tested with Unsloth yet. Worth trying `FastLanguageModel.from_pretrained("Zyphra/ZAYA1-8B")` when GPU is available.
- Double quantization: added May 2026 — saves ~0.36 GB per QLoRA paper benchmarks (was the difference between OOM and success on 16GB GPUs).

### Upstream boundaries (DO NOT CHANGE)
These ZAYA innovations must not be touched:
- CCA attention (compressed convolutional attention)
- MOD skip expert (Mixture-of-Depths)
- EDA routing (depth-wise averaging)
- Fused bias+SwiGLU custom autograd
- FP32 residual accumulation
- Dual time-stream values (val_proj1 + val_proj2)

### Quality gates for training data
1. Mechanical verify hook (exit_code=0)
2. Jaccard tool selection ≥0.7
3. Zero dangerous command flags
4. Zero schema validation errors
5. Token budget ≤4096

### File responsibilities
| File | Role | Change policy |
|------|------|---------------|
| `scripts/train.py` | QLoRA SFT via TRL SFTTrainer | Uses SFTConfig, assistant_only_loss=True |
| `scripts/train_grpo.py` | GRPO policy improvement | Uses GRPOConfig, dapo loss, vLLM colocate |
| `scripts/serve_zaya1.py` | vLLM inference server for ZAYA1-8B | Matches official Zyphra deployment, FP8/MXFP4 quant support, `--enforce-eager` |
| `scripts/convert_zaya_to_gguf.py` | HF safetensors → FP16 GGUF converter | Shortens names to fit GGUF 64-char limit. Produces name_map.json. |
| `scripts/fix_gguf_arch.py` | Rewrite GGUF architecture field | Tool for fixing GGUF arch string in binary |
| `scripts/remap_to_zaya.py` | Godspeed JSONL → ZAYA ChatML | Must preserve `<zyphra_tool_call>` format |
| `scripts/mutate_tasks.py` | 200+ variant tasks | 6 mutation types, 30% OOD minimum |
| `scripts/wsl_fix_moe_scale_routing.py` | Fix scale→FusedMoE param routing in zaya.py | Applies to vLLM WSL install; prevents `quant_method` ValueError. Session 1. |
| `scripts/wsl_fix_marlin_group_size.py` | Skip Marlin repack for group_size≠16 Linear layers | Applies to vLLM WSL install; enables Python dequant fallback. Session 1. |
| `scripts/wsl_fix_nvfp4_text_gen.py` | Three text-generation fixes (gate/up split, lm_head dequant, Path A MoE) | Applies to vLLM WSL install AFTER session-1 fixes. Required for coherent output. |
| `scripts/apply_professional_fixes.py` | Apply 7 production vLLM patches for CT Zaya | Run after vLLM install; includes input_quant guards, CCA calculators, registry |
| `scripts/wsl_run_smoke.sh` | Smoke test: load NVFP4 CT model via vLLM | Expect exit 0; verifies weight loading + inference init |
| `scripts/wsl_run_quick_check.sh` | Generation smoke: raw + chat prompt, dtype=bf16 | Expect " Paris..." for raw, BST explanation for chat. Verifies coherent output. |
| `scripts/wsl_quick_check.py` | Python harness for `wsl_run_quick_check.sh` | Uses Zyphra/ZAYA1-8B tokenizer, prints token ids + decoded text |
| `data/generate.py` | Godspeed → ChatML extraction | Legacy — prefer remap_to_zaya.py |
| `configs/lora_tool_call.yaml` | Single source of truth for hyperparams | rsLoRA, chunked_nll, Liger Kernel, double quant |
| `patches/` | Runtime monkey-patches + vLLM plugins | `apply_zaya_patches.py` auto-runs in train.py/train_grpo.py |
| `tests/` | 100 unit tests | Run before any commit |

### NVFP4 quantization pipeline

**GGUF path** (Phase 1 — COMPLETE):
- **GGUF converter**: `scripts/convert_zaya_to_gguf.py --arch llama` (llama arch for llama.cpp quantize compat)
- **NVFP4 quantizer**: llama.cpp `build/bin/llama-quantize input.gguf output.gguf NVFP4`
- **NVFP4 fallback fix**: Added `GGML_TYPE_NVFP4 → GGML_TYPE_F16` case in `llama-quant.cpp` line 391 (submitted to upstream)
- **Result**: 4.76 GB NVFP4 ZAYA1-8B at 4.52 bpw, 1641 NVFP4 + 842 F16 tensors, BPE tokenizer embedded
- **Output**: `/tmp/zaya1-8b-nvfp4-tok.gguf` (4.67 GB with tokenizer) + `/tmp/zaya1-8b-nvfp4.name_map.json` (2,483 entries)
- **Weights verified**: 0.026 mean error vs original bf16 model (expected 4-bit noise)
- **Status**: GGUF built and verified. vLLM GGUF handler path blocked — requires NVFP4 tensor type support unavailable in vLLM 0.20.2

**Compressed-tensors path** (Phase 2 — Stage 1 COMPLETE, Stage 2 NEXT):
- **Stage 1** ✅ (2026-05-14, sessions 1 + 2): Quantized ZAYA1-8B BF16 → compressed-tensors NVFP4 (`zaya1-8b-nvfp4-ct-gs16/`, group_size=16, 5.04 GB). **Model loads and produces coherent text via vLLM**: "The capital of France is" → " Paris.", BST explanation coherent. 4,244/4,244 weights loaded, 5.53 GiB VRAM, ~0.86 tok/s on Path A Python dequant.
  - **Inference contract**: `dtype="bfloat16"` is required. fp16 collapses output to a single repeated token because Python MoE dequant accumulation precision is insufficient at fp16.
  - **5 vLLM patches required** (applied to WSL install at `/home/ttimm/vllm-env/`, all idempotent):
    1. `wsl_fix_moe_scale_routing.py` — Routes `weight_scale` checkpoint keys to correct FusedMoE params (`w13_weight_scale`/`w2_weight_scale`) instead of weight params, preventing `ValueError: quant method must be one of ['tensor', 'channel', 'group', 'block']` at `fused_moe/layer.py:1359`.
    2. `wsl_fix_marlin_group_size.py` — Checks `self.group_size` against `FP4_MARLIN_SUPPORTED_GROUP_SIZES=[16]` before calling `prepare_fp4_layer_for_marlin()`. Skips Marlin Linear repack on unsupported group sizes and falls back to Python dequant.
    3. `wsl_fix_nvfp4_text_gen.py` fix #1 — Splits combined `linear_fc1.weight_packed` AND `linear_fc1.weight_scale` into gate (`shard_id="w1"`) + up (`shard_id="w3"`) halves on load. `FusedMoE._load_w13` narrows by half when `is_act_and_mul=True`, so passing the full combined tensor with `shard_id="w1"` only loaded gate; up rows stayed at `torch.empty` initialization. The "combined w13_weight_scale" fast-path in `fused_moe/layer.py:1298` is gated on `"ModelOpt" in quant_method_name`, so CompressedTensors fell through to the same narrowing bug for scales too.
    4. `wsl_fix_nvfp4_text_gen.py` fix #2 — Buffers `lm_head.weight_packed` + `lm_head.weight_scale` during load, dequantizes via `compressed_tensors`, and binds into `model.embed_tokens.weight` (canonical tied-embedding name when `tie_word_embeddings=True`). Default loader silently skipped both NVFP4 keys because `ParallelLMHead(quant_config=None)` only registers `lm_head.weight` (unquantized). The skip was masked by a broken log f-string at zaya.py:1037 (missing `f` prefix logged the literal `{chkpt_weight_name}`); the patch also fixes the format string.
    5. `wsl_fix_nvfp4_text_gen.py` fix #3 — Rewrites `CompressedTensorsW4A4Nvfp4MoEMethod` for **Path A**: keep packed FP4 weights + per-group scales, dequant on the fly in `apply()` via `unpack_fp4_from_uint8` + `dequantize`, manual per-expert SwiGLU loop (`silu(first_half) * second_half`, vLLM SiluAndMul convention). Bypasses both Marlin MoE (which corrupts scales via FP8→S0E5M3 sign-flip at `marlin_utils_fp4.py:108-112` for this checkpoint) and the WSL-device-mismatched emulation backend. `fused_experts` returns zero output for constant-routed warmup batches on uncached sm_120 Triton configs, hence the manual loop.
  - **Dequant strategy** (Path A):

    | Layer type | Method | Where |
    |-----------|--------|-------|
    | MoE experts | On-the-fly Python dequant, manual SwiGLU loop | `compressed_tensors_moe_w4a4_nvfp4.py apply()` |
    | CCA attention (linear_q/k, o_proj, val_proj1/2) | Python dequant fallback | `compressed_tensors_w4a16_nvfp4.py apply_weights()` |
    | Router projections | Unquantized fp16 (no dequant) | bf16 `weight` key in checkpoint, loaded directly |
    | lm_head (tied with embed_tokens) | Dequantized once at load, stored as fp16 weight | `zaya.py load_weights` finalization |
- **Stage 2** (24-37 hrs): Custom Blackwell NVFP4 Tensor Core CUDA kernel as drop-in replacement for both the MoE Path A dequant and the Linear Python fallback. ~4-5 GB VRAM, sm_120 hardware-accelerated. Reusable across all NVFP4 CT models. Should drop the bf16-required contract (higher-precision Tensor Core accumulation registers).
- **Config**: `quantization_config: {quant_method: "compressed-tensors", format: "pack-quantized", config_groups: {group_0: {weights: {num_bits: 4, type: "float", strategy: "group", group_size: 16, symmetric: true, scale_dtype: "float8_e4m3fn"}, targets: ["Linear"]}}, ignore: ["router"]}`

**Key technical references**:
- `CompressedTensorsW4A16Fp4` at `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py`
- `CompressedTensorsConfig` at `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`
- NVFP4A16 scheme at `compressed_tensors/src/compressed_tensors/quantization/quant_scheme.py`
- FP4_E2M1_DATA at `compressed_tensors/src/compressed_tensors/quantization/quant_args.py`
- Weight packing: `weight_packed` (uint8, 2x 4-bit per byte), `weight_scale` (FP8_E4M3 per group of 16), `weight_global_scale` (per-output-channel FP32)

## Development workflow
1. `uv sync --dev` to install deps
2. `uv run ruff check .` for linting (0 errors required)
3. `uv run pytest tests/` for tests (100 required, 0 failures)

## Engineering Standards (Non-Negotiable)

### SOTA Only
Every line of code, every architectural decision, every configuration value must be
state-of-the-art for May 2026. No legacy patterns, no "good enough" shortcuts.
When two approaches exist, benchmark both and take the better one.

### Documentation-Backed
No claim, magic number, or design choice stands without a verifiable source. Every
hardcoded value, every hyperparameter default, every model reference must be traceable
to one of:
- Primary: peer-reviewed paper (arXiv, conference), official model card, upstream README
- Secondary: published benchmark result, authoritative blog post, verified issue/PR
- Never: LLM conjecture, unverified StackOverflow, "it works on my machine"

Reference format in code comments: `(ref: arXiv XXXX.XXXXX §3.2)` or `(ref: HF model card 2026-05-11)`.

### Professional Code
- `from __future__ import annotations` in every `.py` file
- Type hints on all public functions (no `Any` unless truly dynamic)
- Ruff: 0 errors (E, F, I, N, W, UP rules)
- No bare `except:`, no `except Exception:` without logging
- Logging: `logging.getLogger(__name__)` with structured messages
- Pytest: 100% pass rate before any commit
- Secrets: never committed, use environment variables, `.gitignore`-protected `.env`

### Verification Protocol
Before claiming something works:
1. Run the thing end-to-end (not just unit tests — integration verification)
2. Check against the source documentation for correctness
3. If training: run a dry-run batch (forward + backward pass) before full run
4. If deployment: verify health endpoint responds before reporting success

### 🔴 FIRST: CUDA graph capture is numerically broken on SM120

**Before diagnosing ANY output or performance problem, rule this out.**
CUDA graph capture — not any specific kernel — produces numerically wrong
output for this model on consumer Blackwell (SM120). Confirmed 2026-08-14 by
sweeping three independent backends (`flashinfer_cutlass` default, `cutlass`,
and `marlin`) all under graph capture: all three produced garbage. The same
weights with `enforce_eager=True` produced coherent, on-topic output on every
backend. `marlin` is the decisive datapoint — it is weight-only and barely
touches the FP4 MoE path, and it still failed identically. **There is no
`--moe-backend` flag that fixes this** — an earlier version of this note
recommended `marlin` as a workaround; that was wrong, corrected here.

**The only fix: `enforce_eager=True`.** Correct, still genuinely W4A4,
forfeits the graph speedup. See `README.md` → "Known Issue: CUDA graph
capture corrupts output on SM120" and `RESEARCH.md` §5.14 for the full
backend sweep and root-cause writeup.

Symptom cluster — any of these means suspect graph capture, not the checkpoint:
- garbage or empty completions, **worse at greedy than at sampling**
- loglikelihood evals scoring fine (ranking-only tasks can't detect this —
  see [[gotcha_zaya_benchmark_chat_template]] equivalent note in this repo)
- throughput wildly nondeterministic across launches on an identical command

**Diagnostic rule: before concluding a checkpoint is damaged, re-run one
prompt with `enforce_eager=True`.** Thirty seconds. Months of "is the
checkpoint healthy" work proceeded without it.

### Benchmarking Protocol (every rule here was learned by breaking it)

**Batch-1 decode throughput on this hardware is not reproducible from a
single run.** Once the CUDA-graph correctness bug above is controlled for
(`enforce_eager=True`), remaining run-to-run variance is small (3.6–3.9%,
see `RESEARCH.md` §5.14) — but before that fix was known, an identical
command measured 9.6 / 32.2 / 23.7 tok/s across three invocations, each
internally tight to 0.2–1.8%. **A 3.4× range**, symptomatic of the
correctness bug itself rather than ordinary noise.

1. **Never publish a throughput number from one run.** Median of ≥5 process
   invocations, and report the range. Within-run spread is *not* evidence of
   reliability — a run can spread 0.2% internally and still land 3× off the
   next invocation.
2. **To reproduce a published number, run the published command.** Running a
   different workload and comparing is a different experiment, not a failed
   reproduction. This mistake produced a false "does not reproduce" conclusion
   earlier in this project's history.
3. **Close GPU consumers first.** Wallpaper Engine, browsers, Steam. An
   animated wallpaper polls at 0–1% utilisation while still inflating
   run-to-run spread from 0.2% to 181%. **Instantaneous utilisation does not
   detect it** — guard on the measured spread instead, and reject any run
   spreading >3%. (`~/scripts/we-control.sh pause|play` handles this.)
4. **Warm the compile cache in a separate, unmeasured stage.** Cold was
   12m31s vs 2m43s warm; a cold-cache timing measures the CUDA compiler, not
   the model.
5. **Judge success by artifact content, not exit code.** vLLM can abort at
   teardown *after* writing a valid result. Validate the JSON parses — do not
   apply a size threshold (a 200-byte floor once rejected a valid 188-byte
   result).
6. **Interleave A/B reps** (A,B,A,B…), never blocked. Blocked runs let
   thermal drift and run order masquerade as the effect being measured.
7. **Accuracy comparisons between checkpoints are PAIRED.** Use
   `--log_samples`, join per `doc_id`, verify `doc_hash` matches so both runs
   provably scored the same items, then McNemar on discordant pairs.
   Comparing aggregate accuracies discards the pairing and wastes the run.
8. **Record the environment fingerprint per run** — vLLM *commit* (this is a
   source build; the version string is insufficient), flashinfer, torch,
   driver, CUDA. A stack change with unchanged throughput still invalidates
   prior claims.
