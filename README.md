# zaya1-nvfp4-w4a4

[![lint](https://img.shields.io/github/actions/workflow/status/t-timms/zaya1-nvfp4-w4a4/lint.yml?style=flat-square&label=lint)](https://github.com/t-timms/zaya1-nvfp4-w4a4/actions/workflows/lint.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![vLLM](https://img.shields.io/badge/vLLM-0.20.2%20source%20build-1E90FF?style=flat-square)](https://github.com/vllm-project/vllm)
[![CUTLASS](https://img.shields.io/badge/CUTLASS-4.4.2%20SM120-76B900?style=flat-square)](https://github.com/NVIDIA/cutlass)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)](./LICENSE)

**NVFP4 W4A4 inference for Zyphra's ZAYA1-8B on consumer Blackwell (RTX 5070 Ti, SM120)** —
4-bit weights *and* 4-bit activations running on native CUTLASS FP4 tensor-core kernels,
at **9.5 tok/s single-stream / ~74 tok/s batch-8** (`enforce_eager=True`; within a
16 GB VRAM budget). An earlier **102.6 / 407.4 tok/s** figure was measured under
CUDA graphs, a code path since confirmed to corrupt output on this card — see
[Known Issue](#known-issue-cuda-graph-capture-corrupts-output-on-sm120), retracted
2026-08-14.

Two checkpoints are published: a **6.02 GB** fully-uniform build and a **9.46 GB**
mixed-precision build. The difference between them has been measured with a paired
test over 14,319 items — see [Checkpoints](#checkpoints).

**Generative accuracy (6.02 GB uniform, 2026-08-16):** **HumanEval 72.6%** pass@1
(95% CI [65.3, 78.8]), **GSM8K 65.5%** [62.9, 68.0], **MMLU-Pro 48.1%** [44.5, 51.8]
(0-shot). Published figures put Qwen 3 7B at ~68–72% and Llama 3 8B at 62–65% on
HumanEval *at full precision* — this matches or beats them at 4-bit weights **and**
activations. Zyphra publishes no HumanEval number for ZAYA1, so this appears to be
the first at any precision. **MMLU-Pro here is 0-shot and is not comparable to
Zyphra's 5-shot 74.2%**; that gap is a protocol difference, not quantization damage
(published INT4 loss on MMLU-Pro is ~1.6 pp). Reasoning budget was tested, not
assumed: doubling it to 8192 moved GSM8K +0.15 pp (p=0.96) and MMLU-Pro +3.29 pp
(p=0.067) — neither significant. Full analysis: [`RESEARCH.md` §5.22](./RESEARCH.md).

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
| **9.5 tok/s** single / **~74 tok/s** batch-8 | `enforce_eager=True` on RTX 5070 Ti — CUDA graphs produce corrupted output on this card, see [Known Issue](#known-issue-cuda-graph-capture-corrupts-output-on-sm120) |
| **6.02 GB / 9.46 GB checkpoints** | Uniform build: all 1,320 Linears packed NVFP4 W4A4, zero BF16 exemptions. Mixed build: 936 W4A4 + 384 outlier-sensitive Linears kept BF16. The exemptions cost 3.44 GB and buy 0.71 pp of HellaSwag |
| **−36% size for −0.71 pp** | Removing all BF16 exemptions costs 0.71 pp on HellaSwag (n=10,042, 95% CI [−1.26, −0.15], paired McNemar). Measured, not assumed — the first attempt used an unpaired test and got the sign wrong |
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

## Checkpoints

| | [`-uniform`](https://huggingface.co/Ttimms/zaya1-8b-nvfp4-w4a4-uniform) | [`zaya1-8b-nvfp4-w4a4`](https://huggingface.co/Ttimms/zaya1-8b-nvfp4-w4a4) |
|---|---:|---:|
| Weights (`model.safetensors`) | **5.99 GB** | 9.46 GB |
| Repository total | **6.02 GB** | 9.49 GB |
| W4A4 Linears | 1,320 | 936 |
| BF16-exempted | **0** | 384 |
| HellaSwag `acc` (n=10,042) | 45.79% | **46.49%** |
| HellaSwag `acc_norm` | 60.65% | **61.34%** |
| KV cache on 16 GB | **6.83 GiB / ~336k tok** | materially less |

**Use `-uniform`** for maximum KV/context headroom or batching. **Use the 9.46 GB
build** for best measured accuracy, or to reproduce the control below.

> **On the two size figures.** `6.02 GB` is the uniform build's **whole repository**;
> `9.46 GB` is the mixed build's **`model.safetensors` alone** (its repository total
> is 9.49 GB). Both figures are individually correct and both are already published,
> but they measure different things — so the table above gives each basis
> explicitly. The reduction holds either way: **−36.7%** on weights, **−36.6%** on
> repository totals. Measured from the Hub, 2026-08-09.


### Why the exemptions were removable

Only **24** `linear_fc2` modules have calibrated activation `max_abs > 500` (worst:
8,896 at `L75.experts.1.linear_fc2`, 622× the median). But **FusedMoE requires
uniform quantization per layer**, so protecting them forced exempting `fc1` *and*
`fc2` across all 16 experts in each affected layer — **384 Linears, 3.44 GB.** A
16× overpay, where 16 is `num_experts`.

They turned out to be largely redundant: SOAR targets the same FP8 block-scale
rounding error and landed *after* the mixed-precision decision, so the two
mitigations were never re-tested together. No residual correction (ARCQuant or
otherwise) is applied to the uniform checkpoint, and none is required.

### Known Issue: CUDA graph capture corrupts output on SM120

**The 102.6 / 407.4 tok/s figures previously published here were measured with
CUDA graphs enabled. That code path is now confirmed to produce numerically wrong
output on this card. The figures are retracted — not revised, retracted.**

A backend sweep on `zaya1-8b-nvfp4-w4a4-uniform` (greedy decode, identical prompt,
weights, and tokenizer across rows) found CUDA graph capture corrupts generation
regardless of MoE kernel:

| backend | with CUDA graphs |
|---|---|
| `flashinfer_cutlass` (default) | garbage |
| `cutlass` | garbage |
| `marlin` (weight-only) | garbage |
| any backend, `enforce_eager=True` | ✅ coherent |

Marlin is weight-only and barely touches the FP4 MoE path — it failing exactly
like the native FP4 kernels rules out a kernel-specific bug. **The fault is graph
capture itself, not any one kernel.**

Two upstream issues are adjacent but neither currently fixes this exact failure
mode. [CUTLASS #3096](https://github.com/NVIDIA/cutlass/issues/3096) fixes a
*different*, non-graph-capture grouped-GEMM correctness bug (stalled, no
confirmed upstream merge). [FlashInfer #2776](https://github.com/flashinfer-ai/flashinfer/issues/2776)
is graph-capture-specific but attributes the corruption to a workspace-buffer
memory-alignment bug in one FlashInfer TRTLLM kernel — that would not explain
Marlin failing too, so this project's finding is narrower than what either issue
documents, and may be worth filing separately. Backend-dependent correctness
drift is a documented class of problem in inference serving generally, not
unique to this stack — see Pape, Evertz & Schönherr,
["The Silent Hyperparameter: Quantifying the Impact of Inference Backends on LLM
Reproducibility"](https://arxiv.org/abs/2605.19537) (arXiv:2605.19537, 2026).

**Only `enforce_eager=True` is confirmed correct on this checkpoint.** Re-verified
2026-08-14 via chat-templated generation (not just the throughput benchmark,
which measures speed on synthetic tokens and would not itself catch this) —
on-topic, factually grounded chain-of-thought reasoning on all three test
prompts, zero garbage or repetition artifacts. An earlier raw-completion test
(no chat template) produced fluent-but-off-topic text; that was a test-harness
error — this model needs its chat template for generative output regardless of
backend — not evidence of a second bug.

### Throughput and memory (measured, `enforce_eager=True`)

Measured with vLLM's own benchmark CLI, not a bespoke harness. **5 separate
process invocations per configuration**, not repeated iterations inside one
process — that's where inference-serving variance actually lives (Kalibera &
Jones, *Rigorous Benchmarking in Reasonable Time*):

```bash
vllm bench latency --model <checkpoint> --dtype bfloat16 --kv-cache-dtype fp8 \
  --gpu-memory-utilization <frac> --max-model-len 4096 --no-enable-prefix-caching \
  --input-len 128 --output-len 256 --batch-size <1|8> --enforce-eager \
  --num-iters-warmup 1 --num-iters 1
```

| | 6.02 GB uniform | 9.46 GB mixed |
|---|---:|---:|
| Single-stream (batch-1), median | **9.52 tok/s** | **9.51 tok/s** |
| Single-stream, range (5 reps) | 9.48–9.84 tok/s | 9.45–9.81 tok/s |
| Batch-8, median (aggregate) | **73.4 tok/s** | **74.4 tok/s** |
| Batch-8, range (5 reps) | 72.2–74.9 tok/s | 72.8–75.7 tok/s |
| Batch-8 scaling vs. batch-1 | 7.71× (96% of ideal 8×) | 7.82× (98% of ideal 8×) |
| `--gpu-memory-utilization` | 0.85 | 0.92 |

**Throughput is still unchanged between checkpoints** — uniform and mixed land
within run-to-run noise of each other at both batch sizes. That part of the
original claim holds; only the absolute numbers were wrong.

**Batching is near-ideal, and that's a real finding, not just a corrected
number.** 96–98% of theoretical linear scaling on a batch of 8 means the MoE
decode step's per-step cost is not growing meaningfully with batch size —
consistent with the earlier hypothesis (`RESEARCH.md`) that a MoE GEMM kernel
falls back to a batch-independent generic tactic rather than a batch-aware one.

**Variance dropped by roughly two orders of magnitude** once CUDA graphs were
disabled — 3.6–3.9% spread across 5 independent process launches here, versus a
previously documented 3.4× (340%) run-to-run swing under the CUDA-graph path.
That instability was itself a symptom of this bug, not unrelated noise.

**The two checkpoints cannot be run at the same memory fraction on a 16 GB
card.** The mixed checkpoint fails at `--gpu-memory-utilization 0.85` with
`ValueError: No available memory for the cache blocks`, and at `1.0` with
`Free memory on device cuda:0 (14.66/15.92 GiB) ... less than desired` because a
desktop session holds ~1.3 GiB. Its working range is narrow; the uniform build
runs at 0.85 with headroom — the practical case for the smaller checkpoint, more
so than the accuracy difference.

> **Conditions.** GPU otherwise idle — Wallpaper Engine and browser GPU
> acceleration closed before measuring; Wallpaper Engine alone previously
> produced a 181% latency spread in this repo's benchmarks. `enforce_eager`
> skips compile/graph-capture warmup entirely, so "cold vs. warm cache" does not
> apply here the way it did under CUDA graphs. Measured 2026-08-14.

**Retracted, do not cite:** 102.6 tok/s single-stream / 407.4 tok/s batch-8
(CUDA graphs; coherent output was never verified at that speed).

> **An unverified external signal exists and disagrees with the table above.**
> llama.cpp [PR #23112](https://github.com/ggml-org/llama.cpp/pull/23112)'s
> own author reports 45.9 tok/s on an RTX 4070 Ti (Q4_K_M GGUF) — a slower GPU
> beating this checkpoint's 9.5 tok/s by ~4.8×. Five independent attempts to
> reproduce it directly on this project's own SM120 hardware all hit the same
> non-deterministic hang (works once, hangs on an identical rerun) — ruled out
> the model, the CUDA toolkit version, and the llama.cpp commit as causes; the
> fingerprint matches known WSL2/Blackwell driver-level deadlock reports. Not
> resolved as of 2026-08-14. Full six-attempt diagnostic log: `RESEARCH.md` §5.16.
>
> **Update, same day:** the gap likely has a principled cause independent of
> the hang above. Activation quantization (W4A4) gives no speed benefit at
> batch-1 — decode there is memory-bandwidth-bound, and quantizing
> activations only helps when compute is the bottleneck (batched serving,
> prefill). Weight-only quantization is expected to win at batch-1 by design;
> W4A4's advantage is memory footprint and batched throughput, which this
> project already measures at batch-8 (73–74 tok/s, 96–98% of ideal scaling).
> Detail: `RESEARCH.md` §5.17.
>
> **A real batch-1 lever that does exist:** vLLM's built-in n-gram speculative
> decoding gives a validated **2.2× speedup on coding-edit prompts** (zero
> training, one config flag) — no gain on free-form generation, as expected,
> but coding-edit is most of an agentic coding workload. **Deployed to the
> production serve script and validated live** through the real
> OpenAI-compatible API, not just the offline benchmark harness. Detail:
> `RESEARCH.md` §5.18.

### Paired evaluation

Exact-binomial **McNemar on discordant items**, joined per `doc_id`, 14,319 items
per checkpoint, four pure-loglikelihood tasks. No generation, so this is immune to
the `<think>`-never-terminates artifact; no chat template, since these are
ranked-continuation tasks.

| task | metric | n | 6.02 GB | 9.46 GB | Δ pp | 95% CI | p |
|---|---|---:|---:|---:|---:|---|---:|
| hellaswag | acc | 10,042 | 45.79% | 46.49% | **−0.71** | [−1.26, −0.15] | 0.0140 |
| hellaswag | acc_norm | 10,042 | 60.65% | 61.34% | −0.70 | [−1.39, −0.01] | 0.0504 |
| arc_challenge | acc | 1,172 | 37.97% | 36.95% | +1.02 | [−1.42, +3.47] | 0.4522 |
| arc_challenge | acc_norm | 1,172 | 37.97% | 40.36% | −2.39 | [−4.97, +0.19] | 0.0799 |
| winogrande | acc | 1,267 | 56.20% | 59.04% | −2.84 | [−6.04, +0.36] | 0.0906 |
| piqa | acc | 1,838 | 69.42% | 70.02% | −0.60 | [−2.41, +1.21] | 0.5564 |
| piqa | acc_norm | 1,838 | 70.89% | 70.08% | +0.82 | [−1.02, +2.65] | 0.4166 |

> **Read the intervals, not the p-values.** Nothing survives Bonferroni
> (α = 0.05/7 = 0.0071) — but that is absence of *resolution*, not evidence of
> absence. HellaSwag is the only adequately powered task and its CI excludes zero;
> the smaller benchmarks still admit −4.97 pp (arc_challenge `acc_norm`) and
> −6.04 pp (winogrande). Five of seven comparisons point negative.
>
> **Defensible claim: −0.71 pp on HellaSwag for −36% size.** Nothing stronger.

The first attempt at this comparison used ARC-Easy and an **unpaired**
two-proportion test, which reported +1.81 pp in favour of the smaller checkpoint
at p=0.18. Both checkpoints are quantizations of one base model scored on the same
items; discarding that pairing discards the power. Aggregate accuracy output
cannot be converted into a paired test after the fact — it needs per-item logging
(`log_samples=True`) from the start.

Reproduce: `scripts/phase_a_driver.sh` (runs both checkpoints and resumes), or
`scripts/run_phase_a.py` + `scripts/analyze_phase_a.py` individually.

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
zaya1-nvfp4-w4a4/
├── scripts/
│   ├── quantize_zaya_ct_nvfp4.py       # NVFP4 W4A4 quantization + layer-wise activation calibration
│   ├── build_calibration_data.py       # Calibration mix (incl. ARC/HellaSwag phase-2 mix)
│   ├── fix_w4a4_global_scales.py       # Post-hoc global-scale convention repair
│   ├── verify_w4a4_dequant.py          # Round-trip dequant vs HF reference (≈1% rel. error)
│   ├── run_full_benchmarks.py          # lm-eval suite w/ chat template (loglikelihood tasks; CUDA graphs are broken for generation, see Known Issue)
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
bind, the bf16-only inference contract, the global-scale convention, the
CUDA-graph memory-profiler flag that silently eats 3.5 GB on a 16 GB card, and
the CUDA graph *capture* bug that corrupts generation outright (see
[Known Issue](#known-issue-cuda-graph-capture-corrupts-output-on-sm120)).

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
