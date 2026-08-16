# Consumer Blackwell Deployment of ZAYA1-8B: W4A4 NVFP4 Quantization, Three Silent-Corruption Failures, and What the Model Is Actually For

**Tremayne Timms**, ttimmsinternational@gmail.com
*Independent research. First draft May 2026. Substantially corrected August 2026.*

---

> ## ⚠️ Corrigendum, 2026-08-16
>
> **The first version of this paper (May 2026) reported 102.6 tok/s single-stream
> and 407.4 tok/s at batch-8, and recommended CUDA graph capture as the serving
> configuration. Both claims were wrong, and the second is dangerous.**
>
> CUDA graph capture produces numerically incorrect output for this model on
> SM120. The failure is silent: the model emits fluent, confident text that is
> unrelated to the prompt. Every throughput and accuracy number in the original
> paper was measured on that path and is therefore void.
>
> | Original claim | Corrected |
> |---|---|
> | 102.6 tok/s single-stream | **9.5 tok/s** (`enforce_eager=True`) |
> | 407.4 tok/s batch-8 | **~74 tok/s** |
> | "12.8x speedup from CUDA graphs" | Withdrawn. Eager is the only correct mode |
> | Checkpoint 8.9 GiB | **9.46 GB** mixed, **6.02 GB** uniform |
> | ARC-Easy 68.6%, HellaSwag 61.5% | Void, re-measured in §6 |
>
> Anyone who deployed using the original paper's configuration was getting
> corrupted output. That is the reason this correction is prominent rather than
> a footnote.
>
> The correction is also the most interesting result in this work. Three
> independent failures in this project all shared the same signature: a clean
> exit code, plausible-looking output, and completely invalid results (§7.6).

---

## Abstract

ZAYA1-8B (Zyphra, 2026) reaches strong reasoning benchmarks with 760M active
parameters via top-1 mixture-of-experts routing, but its 17.7 GB BF16 footprint
excludes consumer GPUs. We present a W4A4 NVFP4 quantization of ZAYA1-8B for the
RTX 5070 Ti (Blackwell SM120), quantizing 4-bit weights *and* 4-bit activations,
and publish two checkpoints: a 9.46 GB mixed-precision build and a 6.02 GB fully
uniform build with no BF16 exemptions. A paired McNemar test over 14,319 items
bounds the accuracy difference between them at -0.71 pp HellaSwag.

Measured throughput is 9.5 tok/s single-stream and roughly 74 tok/s at batch-8,
the latter at 96 to 98 percent of ideal scaling. Single-stream is slower than
weight-only 4-bit quantizations of the same model, which is expected: decode at
batch-1 is memory-bandwidth-bound, so quantizing activations adds work without
relieving the bottleneck. W4A4's advantage is memory footprint and batched
throughput, not latency.

On generative benchmarks the 6.02 GB checkpoint scores 72.6% pass@1 on HumanEval
(95% CI [65.3, 78.8]), 65.5% on GSM8K, and 48.1% on MMLU-Pro at 0-shot. The
HumanEval figure appears to be the first published for ZAYA1-8B at any precision,
and it matches or exceeds published full-precision results for Qwen 3 7B and
Llama 3 8B.

We also quantify the reasoning and latency tradeoff directly. Disabling the
model's reasoning trace via its own chat template flag gives roughly an 8.5x
speedup but costs 17 to 29 accuracy points across three benchmarks, all at
p<0.0001. ZAYA1's accuracy is inseparable from its reasoning, and its reasoning
is what makes it slow, so it is not a viable interactive coding assistant in any
quantization format. Its real fit is batch and asynchronous workloads.

Finally, we document three silent-corruption failures encountered during this
work, each of which completed with exit code 0 and produced artifacts that
passed every structural check while being entirely invalid.

---

## 1. Introduction

Sparse MoE architectures achieve a favourable compute-to-quality ratio by
activating a fraction of parameters per token. ZAYA1-8B takes this to an extreme:
16 experts with top-1 routing yields 760M active parameters from an 8.4B budget.

The gap between quality and deployability remains. At 17.7 GB in BF16, ZAYA1-8B
exceeds every consumer GPU. NVFP4, NVIDIA's block-structured 4-bit float format
with Blackwell tensor core support, offers a direct remedy.

Contributions:

1. **A W4A4 NVFP4 checkpoint of ZAYA1-8B**, quantizing activations as well as
   weights, produced by layer-wise GPU calibration. Two variants are published,
   including a fully uniform 6.02 GB build with zero BF16 exemptions.

2. **Honest measurement of what W4A4 buys and costs.** Single-stream decode is
   slower than weight-only quantization, and we explain why rather than hiding
   it. Batched throughput and footprint are where the scheme pays.

3. **The first generative benchmark results for this model**, with confidence
   intervals, including what appears to be the first HumanEval figure for
   ZAYA1-8B at any precision.

4. **A quantified reasoning and latency tradeoff.** Disabling reasoning is 8.5x
   faster and costs 17 to 29 accuracy points. This settles what the model is
   suitable for.

5. **Three silent-corruption failures, documented in full** (§7.4 to §7.6),
   including the one that invalidated this paper's original results.

---

## 2. Background

### 2.1 ZAYA1-8B architecture

An 80-layer MoE model (in the pre-refactor config) with alternating Compressed
Convolutional Attention (CCA) and MoE blocks.

| Property | Value |
|---|---|
| Total parameters | 8.4B |
| Active parameters | 760M |
| Experts | 16, top-1 routing |
| Hidden size | 2048 |
| Attention heads | 8, 2 KV heads |
| Vocabulary | 262,272 |
| Max positions | 131,072 |

**Base model revision.** Quantization used `Zyphra/ZAYA1-8B-legacy`, the
pre-refactor 80-layer config. Zyphra later reshaped the checkpoint for an
upstream transformers PR and published it as `Zyphra/ZAYA1-8B` with
`num_hidden_layers: 40` and `layer_types: hybrid`.

We verified these are the same weights rather than a retrain. Fetching
`model.embed_tokens.weight` from both repositories via HTTP range request gives
byte-identical tensors (1,074,266,112 bytes, BF16, shape [262272, 2048]), and
both repositories report an identical aggregate `total_size` of 17,680,978,928
bytes despite the tensor layout changing from 2,483 named tensors to 1,283. Every
differing config field resolves to a clean unit conversion, for example
`num_hidden_layers` 80 to 40 and `ffn_hidden_size` 4096 to 2048 are both exactly
2:1.

### 2.2 NVFP4

NVFP4 is a block-structured 4-bit float with per-block FP8 scales and a per-tensor
global scale. The convention that matters in practice: `global_scale` must be
`2688 / max_abs` in divisor form, and `weight_scale_fp8` must be pre-multiplied by
it. Getting this backwards produces silent NaN or zero logits rather than an
error (§7.1).

---

## 3. Quantization pipeline

### 3.1 Layer-wise GPU calibration

977 calibration samples of 1024 tokens each, drawn from an ARC-weighted mixture.
Calibration runs layer by layer to fit in 16 GB, with forward hooks recording
per-module activation maxima.

`activation_max` is accumulated as a running maximum over every calibration
token, which makes it order-independent and frequency-independent. This detail
matters later (§7.5).

### 3.2 SOAR global-scale optimization

Max-abs scaling maximises FP4 range coverage but does not minimise reconstruction
error. The dominant error source is FP8 block-scale rounding. A 25-point
log-spaced grid search over candidate global scales, minimising weighted rounding
error, improved HellaSwag acc_norm by roughly 0.9 pp. Max-abs remains a candidate
in the search space, so the method can only match or improve on it.

### 3.3 Mixed-precision exemption, and removing it

Twelve MoE layers carry per-expert activation outliers above 500, the worst being
8,896, roughly 622x the median. FusedMoE requires uniform quantization across all
experts in a layer, so protecting 24 offending modules forces 384 Linears to BF16
and costs about 3.5 GB.

We therefore publish both variants. The uniform 6.02 GB build compresses those
layers to W4A4 anyway. A paired exact-binomial McNemar test over 14,319 items
across four loglikelihood tasks bounds the cost of doing so at **-0.71 pp
HellaSwag, 95% CI [-1.26, -0.15]**.

An earlier claim that this cost was "not measurable" came from an underpowered
*unpaired* test and has been withdrawn. Comparing two quantizations of one base
model on the same items is a paired design, and discarding that pairing discards
the statistical power.

---

## 4. Inference infrastructure

### 4.1 vLLM source build for SM120

vLLM 0.20.2 compiled from source with `TORCH_CUDA_ARCH_LIST=12.0`, enabling
`cutlass_scaled_fp4_mm_sm120a` and `cutlass_fp4_group_mm`. No custom CUDA was
written. These kernels exist in the vLLM source tree but are absent from
pre-built wheels, which is the entire reason a source build is required.

### 4.2 CUDA graph capture is broken on SM120, do not enable it

**This section replaces the original paper's §5.3, which recommended the
opposite.**

CUDA graph capture computes numerically incorrect results for this model on
consumer Blackwell. The failure is silent. Under greedy decoding the model
returns fluent text with no relationship to the prompt.

A backend sweep isolates the cause:

| Backend | With CUDA graphs |
|---|---|
| `flashinfer_cutlass` (default) | garbage |
| `cutlass` | garbage |
| `marlin` (weight-only) | garbage |
| Any backend, `enforce_eager=True` | correct |

`marlin` is the decisive datapoint. It is weight-only and barely touches the FP4
MoE path, yet it fails identically. Three architecturally unrelated compute paths
failing the same way under capture, and all succeeding without it, points at
graph capture itself rather than any kernel.

We later ruled out a second axis. vLLM issue #41651 describes an SM120-specific
bug with a close signature (FlashInfer attention plus FP8 KV cache plus CUDA
graphs producing random output on long prompts) with `TRITON_ATTN` as a
workaround. Testing `attention_backend="TRITON_ATTN"` with graphs enabled:
capture succeeded cleanly, 35 of 35 graphs captured, and output was still
garbage. Two independent axes, four kernel combinations, all fail under capture.

**`enforce_eager=True` is mandatory, not a tuning preference.**

### 4.3 Speculative decoding

vLLM's built-in n-gram speculative decoding gives a validated 2.2x speedup on
code-editing prompts (21.11 versus 9.62 tok/s median, 5 repetitions each) and
essentially nothing on free-form generation, which is expected since it relies on
overlap between prompt and output. It is lossless: rejection sampling preserves
the output distribution exactly, unlike the `enforce_eager` requirement, which is
a correctness constraint rather than a speed tradeoff.

One cost surfaces only at serve time: vLLM disables async scheduling when n-gram
speculative decoding is active. That is not accounted for in the 2.2x figure.

---

## 5. Throughput

Measured over 5 independent process invocations per configuration on an idle GPU,
`enforce_eager=True`, reporting median and range. Repetitions go between
invocations rather than inside one run, because that is where the variance lives.

| | uniform (6.02 GB) | mixed (9.46 GB) |
|---|---:|---:|
| Single-stream, median (range) | 9.52 (9.48 to 9.84) tok/s | 9.51 (9.45 to 9.81) tok/s |
| Batch-8, median (range) | 73.4 (72.2 to 74.9) tok/s | 74.4 (72.8 to 75.7) tok/s |
| Batch-8 scaling | 7.71x (96% of ideal) | 7.82x (98% of ideal) |

**On being slower than weight-only quantization.** A community GGUF of this model
reportedly reaches 45.9 tok/s on an RTX 4070 Ti, a slower GPU. That is not a
defect in this work. Decode at batch-1 is memory-bandwidth-bound, so quantizing
activations adds dequantization work without relieving the actual bottleneck.
Weight-only quantization is expected to win at batch-1 by design. W4A4's
advantage is footprint and batched throughput, visible in the near-ideal batch-8
scaling above.

We attempted to reproduce the 45.9 figure directly on matched hardware across
five independent build, version and flag combinations. All hit the same
non-deterministic hang: the same binary and command runs cleanly once and hangs
on an identical rerun, immune to SIGTERM. We ruled out the model, the CUDA
toolkit version and the llama.cpp commit, and filed it upstream
(microsoft/WSL#41361). The comparison therefore remains unverified on matched
hardware.

---

## 6. Accuracy

### 6.1 Loglikelihood tasks, paired

Exact-binomial McNemar on discordant items, joined per `doc_id`, 14,319 items per
checkpoint across four pure-loglikelihood tasks. No generation, so these are
immune to the unterminated-reasoning artifact described below.

| Task | Metric | n | 6.02 GB | 9.46 GB | Δ pp | 95% CI | p |
|---|---|---:|---:|---:|---:|---|---:|
| hellaswag | acc | 10,042 | 45.79% | 46.49% | -0.71 | [-1.26, -0.15] | 0.0140 |
| arc_challenge | acc | 1,172 | 37.97% | 36.95% | +1.02 | [-1.42, +3.47] | 0.4522 |
| winogrande | acc | 1,267 | 56.20% | 59.04% | -2.84 | [-6.04, +0.36] | 0.0906 |
| piqa | acc | 1,838 | 69.42% | 70.02% | -0.60 | [-2.41, +1.21] | 0.5564 |

Only HellaSwag reaches significance. The smaller tasks are underpowered and
cannot rule out larger effects, which we state rather than reading the
non-significant point estimates as findings.

### 6.2 Generative benchmarks

Loglikelihood tasks score pre-written continuations. They measure ranking, never
production. That is a structural blind spot, and this project has been caught by
it: a checkpoint incapable of forming a coherent sentence once scored 61.18% on
HellaSwag. The following measure generation.

Uniform 6.02 GB checkpoint, `enforce_eager=True`, reasoning budget 4096,
temperature 0.6 and top_p 0.95 per Zyphra's published recommendation, seed 42.

| Benchmark | Score | 95% CI | n |
|---|---:|---|---:|
| **HumanEval** | **72.6%** pass@1 | [65.3, 78.8] | 164 |
| **GSM8K** | **65.5%** | [62.9, 68.0] | 1,319 |
| **MMLU-Pro** (0-shot) | **48.1%** | [44.5, 51.8] | 700 |

**HumanEval.** Published comparisons place Qwen 3 7B at roughly 68 to 72 percent
and Llama 3 8B at 62 to 65 percent, both at full precision. This checkpoint
matches or exceeds them while running 4-bit weights and activations in 6.02 GB.
Zyphra publishes no HumanEval figure for ZAYA1, so this appears to be the first
measurement at any precision.

**MMLU-Pro requires a caveat, and it is not quantization damage.** Zyphra reports
74.2% for BF16. Our 48.1% is not comparable: lm-eval's MMLU-Pro task is 5-shot
and this harness is 0-shot, because the standard harness scores unterminated
reasoning traces as the answer. Published INT4 loss on MMLU-Pro runs around 1.6
pp, so a 26 pp quantization cost would be far outside anything documented, and
§6.1 already bounds this checkpoint at -0.71 pp HellaSwag. Zyphra's number also
comes from a private harness with undisclosed generation limits.

**GSM8K is a regression check, not a capability claim.** GSM8K is heavily
contaminated: removing contaminated items has been shown to cost up to 13 pp.
Read 65.5% as evidence that quantization did not break arithmetic reasoning.

### 6.3 Why budget forcing, and why the budget is 4096

ZAYA1's reasoning trace frequently never terminates within a feasible budget.
lm-eval does support reasoning models via `think_end_token`, but it strips
post-hoc: `generation.split(token)[-1]`. When the model never emits the closing
tag, that returns the entire unterminated trace as the answer and scores it. This
is the artifact that put IFEval at 19.8% against an 85.58% reference.

We therefore implement budget *forcing*: generate bounded reasoning, then inject
the closing tag and decode the answer that gets scored. All scoring still uses
lm-eval's own extraction regexes and HF `evaluate`'s sandboxed `code_eval`, so
only the generation protocol is custom.

The budget was determined empirically, not assumed. Doubling it to 8192 and
comparing with paired McNemar on identical items:

| Benchmark | 4096 | 8192 | Δ | 95% CI | p |
|---|---:|---:|---:|---|---:|
| GSM8K | 65.50% | 65.66% | +0.15 pp | [-2.74, +3.04] | 0.9581 |
| MMLU-Pro | 48.14% | 51.43% | +3.29 pp | [-0.22, +6.66] | 0.0673 |

Neither is significant, and 8192 costs roughly 3x the wall time. Ceiling hits
barely moved (GSM8K 78% to 71%): this model keeps thinking regardless of the room
it is given.

---

## 7. Technical findings

### 7.1 The global-scale convention

NVFP4 `global_scale` must be `2688 / max_abs` in divisor form, with
`weight_scale_fp8` pre-multiplied by it. The inverse convention loads without
error and produces NaN or zero logits, which manifests downstream as pad-token
collapse rather than an exception.

### 7.2 FusedMoE uniform quantization constraint

vLLM's FusedMoE requires uniform quantization across all experts in a layer.
Protecting 24 offending Linears therefore costs 384 exemptions and about 3.5 GB,
an overhead 17x larger than a naive per-module estimate would suggest. This is
the structural reason mixed-precision MoE is expensive, and the reason the
uniform checkpoint exists.

### 7.3 Compressed-tensors calibration interference

`apply_quantization_config` silently replaces `Linear.forward` with a
NaN-producing fake-quant path. Plain `nn.Linear.forward` must be restored before
calibrating activation scales.

### 7.4 CUDA graph capture corrupts output (§4.2)

Documented above. This is the failure that invalidated the original paper.

### 7.5 EBSS calibration produced a degenerate corpus

Expert-balanced sample selection is a published idea for MoE calibration:
resample so under-activated experts receive adequate coverage. Our implementation
produced a checkpoint that scored at chance level (HellaSwag acc_norm 60.29% to
25.75%) while exiting cleanly with byte-identical weights.

Isolation took three steps. Sampling `weight_packed`, `weight_scale` and
`weight_global_scale` across both checkpoints showed them byte-identical, leaving
only `input_global_scale` differing, which narrowed the fault to activation
calibration without any re-runs. That value is inflated on 95.8% of 1,320 modules
with a median ratio of 1.68x, implying observed activation maxima roughly 60% of
their true size, which saturates the FP4 range at inference. The cause was the
selection loop: it appended the winning sample but never masked it from later
rounds, yielding a 977-row corpus containing **3 unique rows, one repeated 972
times** (738 unique token ids versus 28,397).

After fixing selection to sample without replacement, coverage was *unchanged*.
That is arithmetic rather than bad luck: selecting N samples without replacement
from a corpus of N is a permutation, and `activation_max` is a running maximum,
an order-independent and frequency-independent statistic. **No reordering of a
fixed corpus can change a maximum.** EBSS is therefore inapplicable to max-based
activation calibration by construction, and we closed it rather than leaving it
as a maybe-retry.

### 7.6 The pattern worth generalising: exit code 0 means nothing

Three independent failures in this project shared one signature. Each completed
with exit code 0. Each produced artifacts that passed every structural check:
correct file sizes, valid manifests, well-formed JSON. Each was entirely invalid.

| Failure | What passed | What was actually wrong |
|---|---|---|
| CUDA graph capture | Clean generation, fluent text | Output unrelated to the prompt |
| EBSS calibration | 6.02 GB checkpoint, valid manifest | Chance-level accuracy |
| Missing `max_model_len` | Every log line correct | Generation cut by context, not budget |

Only measured behaviour caught any of them. The operational rules we now follow:
judge long runs by artifact content rather than exit status; validate that
artifacts *parse* rather than that they exceed a size threshold (a 200-byte floor
once rejected a valid 188-byte result); and never treat a plausible-looking
output as evidence of a working path.

A related harness trap, encountered twice: using
`tokenizer.apply_chat_template()` followed by `llm.generate()` produces fluent
but completely off-topic output on this model, because the template emits
`bos_token` itself and the path risks a doubled BOS. It mimics a corrupted
checkpoint convincingly. `llm.chat()` is correct.

---

## 8. What this model is for

Disabling the reasoning trace via ZAYA1's own chat template flag
(`enable_thinking=False`, which pre-closes the reasoning block) produces a large
speedup and a large accuracy loss. Paired McNemar on identical items:

| Benchmark | Thinking | No thinking | Δ | p | Wall time |
|---|---:|---:|---:|---:|---|
| HumanEval | 72.6% | 43.9% | **-28.66 pp** | <0.0001 | 15 m to 2 m |
| MMLU-Pro | 48.1% | 26.7% | **-21.43 pp** | <0.0001 | 39 m to 4 m |
| GSM8K | 65.5% | 48.1% | **-17.36 pp** | <0.0001 | 63 m to 8 m |

Roughly 8.5x faster overall for 17 to 29 accuracy points. All highly significant,
with no interval near zero, and discordant counts confirming it is not noise (59
versus 12, 203 versus 53, 402 versus 173).

**ZAYA1's accuracy is its reasoning, and its reasoning is what makes it slow. The
two cannot be separated.** No serving configuration turns it into a responsive
interactive assistant, and this holds for any quantization format: even at 45
tok/s a 4096-token reasoning trace still costs roughly 90 seconds per turn.

Where it does fit:

- Batch and asynchronous work. Batch-8 reaches 74 tok/s at 96 to 98 percent of
  ideal scaling in 6.02 GB, so per-item latency is irrelevant and throughput per
  gigabyte is strong.
- Hard single questions where waiting is acceptable.
- Test-time-compute harnesses, which is Zyphra's own framing.
- Serving several concurrent users on one 16 GB card.

Where the reasoning flag is still useful is per-request routing rather than a
global switch. Notably 12, 53 and 173 items respectively were solved *only* with
reasoning disabled, so some tasks are actively hurt by overthinking.

---

## 9. Limitations

- **Single seed.** All accuracy figures come from one seed. The confidence
  intervals reported cover item-sampling uncertainty and say nothing about
  generation stochasticity at temperature 0.6. Multi-seed repetition is the
  correct next step and has not been done.
- **MMLU-Pro is 0-shot and on a 700-item stratified subset** of 12,032, not the
  full set, and not comparable to 5-shot published figures.
- **No local BF16 baseline is possible.** At 17.7 GB the source model does not
  fit in 16 GB, so retention against BF16 cannot be measured on this hardware.
- **The weight-only comparison is unverified** on matched hardware, blocked by
  the llama.cpp hang described in §5.
- **Generative results are from the uniform checkpoint only.** The mixed
  checkpoint was not re-measured on these benchmarks.
- **The reasoning comparison is protocol versus protocol** (two-stage budget
  forcing versus single-stage direct), not a single-variable ablation.

---

## 10. Conclusion

ZAYA1-8B can be quantized to 4-bit weights and 4-bit activations and served on a
16 GB consumer card at 6.02 GB, reaching 72.6% pass@1 on HumanEval, which matches
or exceeds published full-precision results for comparable 7 to 8B models. The
accuracy cost of removing all BF16 exemptions is bounded at -0.71 pp HellaSwag
over 14,319 paired items.

Single-stream throughput of 9.5 tok/s is slower than weight-only quantization of
the same model, for a principled reason rather than an implementation defect, and
the scheme's advantage appears in footprint and batched throughput instead.

The model is not suitable as an interactive coding assistant, and we establish
that with measurement rather than assertion: its accuracy depends on reasoning
that costs 17 to 29 points to remove, and that reasoning is what makes it slow.

The most transferable result is not a number. Three independent failures in this
work produced clean exit codes, valid-looking artifacts, and entirely invalid
results, including the ones that appeared in the first version of this paper.
Serving-stack correctness on new hardware cannot be assumed from the absence of
errors, and published numbers are worth exactly as much as the verification
behind them.

---

## Appendix A: Reproducibility

```bash
# 1. Build vLLM from source with SM120 NVFP4 kernels
cd vllm-src && TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=8 pip install -e . --no-build-isolation

# 2. Quantize
python scripts/build_calibration_data.py --arc-mix
python scripts/quantize_zaya_ct_nvfp4.py --scheme w4a4 --mixed-precision-threshold 500

# 3. Serve (enforce-eager is mandatory for correctness)
bash scripts/serve.sh

# 4. Generative benchmarks
bash scripts/run_budget_forced_suite.sh

# 5. Paired comparison between two runs
python scripts/compare_budget_forced.py <run_a.json> <run_b.json>
```

**Environment.** vLLM 0.20.2 source build (commit `6e2f9c5`), FlashInfer
0.6.8.post1, torch 2.11.0+cu130, driver 610.88, RTX 5070 Ti 16 GB, WSL2.

**Checkpoints.** `Ttimms/zaya1-8b-nvfp4-w4a4` (9.46 GB) and
`Ttimms/zaya1-8b-nvfp4-w4a4-uniform` (6.02 GB) on Hugging Face.

**Code.** https://github.com/t-timms/zaya1-godspeed

---

## References

1. Zyphra. ZAYA1-8B technical report. arXiv:2605.05365.
2. Muennighoff et al. s1: Simple test-time scaling (budget forcing).
3. Biderman et al. Lessons from the Trenches on Reproducible Evaluation of
   Language Models. arXiv:2405.14782.
4. Kalibera and Jones. Rigorous Benchmarking in Reasonable Time.
5. Pape, Evertz and Schönherr. The Silent Hyperparameter. arXiv:2605.19537.
6. NVIDIA CUTLASS issue #3096, SM120 NVFP4 MoE grouped GEMM.
7. vLLM issue #41651, FlashInfer plus FP8 KV cache plus CUDA graphs on SM120.
8. Microsoft WSL issue #41361, non-deterministic CUDA hang on SM120 under WSL2.
