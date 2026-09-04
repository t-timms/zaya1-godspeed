# GPU handoff — ZAYA1 re-quantization and CUDA-graph sweep

*Written 2026-09-03. All non-GPU preparation is done; everything below needs the
GPU. Steps are ordered so the cheapest validation fails first. Do not skip a
validation gate to save time — every gate here exists because something silently
produced a wrong answer once.*

**Background:** RESEARCH.md §5.24 (checkpoints do not load on a current stack) and
§5.17a (decode is dispatch-overhead-bound, not bandwidth-bound). Read both before
starting; the whole plan follows from them.

## State already on disk

| Item | Location | Status |
|---|---|---|
| `-uniform` checkpoint (legacy, **does not load**) | HF cache, 5.7 GiB | fetched |
| `Zyphra/ZAYA1-8B` refactored base (bf16) | HF cache, 17 GB | fetched |
| Config-patched legacy test copy | `~/zaya-configfix-test` | reaches weight load, then fails structurally — evidence only |
| `zaya` MoE registration | `scripts/register_zaya_moe.py` | verified by inspection, **not yet exercised in a real run** |
| Coherence gate + graph sweep | `scripts/check_coherence.py`, `scripts/sweep_cudagraph_modes.sh` | syntax-checked; sweep blocked until a checkpoint loads |

Environments: `~/quant-env` (llmcompressor 0.13.0) for quantization,
`~/vllm-env` (vLLM 0.26.0) for serving/eval. They are **separate** — do not mix.

## Step 0 — rebuild the calibration data (~5-20 min, CPU + network, no GPU)

`data/calibration/` is gitignored and the `.pt` was cleared at some point; only
`manifest.json` survives. The dry run fails in 11 seconds without it:

```
ERROR Calibration data not found: data/calibration/calibration_data.pt.
      Run scripts/build_calibration_data.py first.
```

The published checkpoints used the **8-source mix** recorded in
`data/calibration/manifest.json` (math500 151, gsm8k 153, humaneval 38, mbpp 25,
triviaqa 153, alpaca 153, writingprompts 153, glaive 153 = 979 samples, matching
the "977 samples" on the card). That is the `--arc-mix` recipe:

```bash
cd ~/zaya1-nvfp4-w4a4
ZAYA_MODEL_ID=Zyphra/ZAYA1-8B ~/quant-env/bin/python scripts/build_calibration_data.py --arc-mix
```

Pass `--calibration-data <path>` to the quantizer if the output does not land at
`data/calibration/calibration_data.pt`.

## Step 1 — dry-run the re-quantization (~10 min GPU) ← THE GATE

Registration is now wired into `quantize_zaya_ct_nvfp4.py` itself, immediately
after `parse_args()`, so it lands in the same process as the pipeline. Do not rely
on registering from a separate `python -c` invocation — that patches a different
process and the MoE would silently stay BF16.

```bash
cd ~/zaya1-nvfp4-w4a4
ZAYA_MODEL_ID=Zyphra/ZAYA1-8B ~/quant-env/bin/python scripts/quantize_zaya_ct_nvfp4.py \
    --scheme w4a4 --dry-run \
    --output-dir ~/models/zaya-refactored-w4a4-dryrun
```

Always pass an explicit `--output-dir`; the scheme default points at the directory
the legacy build used.

**The gate — read the module count.** The dry run must report **expert Linears**
among the calibrated modules. If it reports roughly **80** modules, the linearizer
did not engage and the entire MoE would be left in BF16: **stop, do not run the
full job.** The expected order of magnitude with linearization is ~1,300.

`register_zaya_moe` must be imported in the *same process* as the pipeline. If
`quantize_zaya_ct_nvfp4.py` does not import it, add the two-line import at the top
of `main()` rather than patching site-packages.

## Step 2 — full re-quantization (~1–4 h GPU)

Only after Step 1's module count is right.

```bash
ZAYA_MODEL_ID=Zyphra/ZAYA1-8B ~/quant-env/bin/python scripts/quantize_zaya_ct_nvfp4.py \
    --scheme w4a4 --calibration-data data/calibration/arcmix/calibration_data.pt
```

Ignore-list note: `re:.*cca.*` is **dead** on this base (the convs are now under
`qkv_proj` and are `Conv1d` anyway). Keep it or replace it, but do not assume it is
protecting anything. `lm_head`, `re:.*router.*`, `re:.*norm.*`, `re:.*qkv.*` all
still behave — see §5.24.

**Do not overwrite the published artifacts.** Write to a new output dir; the legacy
checkpoints stay as the record behind the published numbers.

## Step 3 — does it load? (~5 min GPU)

```bash
~/vllm-env/bin/python scripts/check_coherence.py \
    --model <new-output-dir> --mode NONE --write-reference
```

This is the whole point of the exercise. It must produce clean output under
`enforce_eager` before anything else happens.

## Step 4 — CUDA-graph sweep (~1 h GPU)

```bash
scripts/sweep_cudagraph_modes.sh <new-output-dir> 0.85
```

Tests `NONE / PIECEWISE / FULL_DECODE_ONLY / FULL / FULL_AND_PIECEWISE`. A mode
failing the coherence gate **cannot** report a throughput number — that is
structural, and it is why the §5.14 retraction happened.

Hypothesis being tested (§5.17a): the corruption is in **CCA state**, not the FP4
MoE kernels — which would explain Marlin failing too — and `PIECEWISE` is the mode
that leaves attention eager. Also re-tests the default, since §5.14 was diagnosed on
vLLM 0.20.2 and this box now runs 0.26.0; it may simply be fixed upstream.

## Step 5 — re-evaluate before publishing anything (several h GPU)

**The published accuracy numbers do not transfer.** They were measured on the legacy
artifact; a re-quantized checkpoint is a new artifact. Re-run before any figure goes
on a card:

- paired HellaSwag/ARC/WinoGrande/PIQA — `scripts/run_phase_a.py` + `analyze_phase_a.py`
- generative suite — `scripts/run_budget_forced_suite.sh`

## Cross-terminal coordination

Only one process should hold the GPU at a time — 16 GB, and the desktop already
holds ~2.5 GB. Before starting, check:

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

Long steps should run detached so they survive the terminal exiting:

```bash
tmux new-session -d -s <name> '<command>'
tmux ls          # check
tmux attach -t <name>
```

Steps 1→2→3 are strictly sequential. Step 4 depends on 3. Step 5 depends on 3 and
should not overlap with 4 — both want the full card.

## Ground rules carried from this project's own history

1. A throughput number without a coherence check beside it is not a result (§5.14).
2. Validate at the smallest scale first — the dry run exists so a bad config costs
   10 minutes, not 4 hours.
3. Under any sampling randomness, repeat 3–5× before comparing.
4. Do not overwrite prior outputs; rename or write to a new directory.

---

## STEP 1 RESULT — 2026-09-03: THE GATE FAILED. DO NOT RUN STEP 2.

The dry run executed and reported:

```
Applying quantization config: 100%|██████████| 80/80
Restored plain Linear.forward on 80 quantized Linears for BF16 calibration
```

**80 Linears.** That is the stop condition: 40 `self_attn.o_proj` + 40
`mlp.gate.down_proj`. The entire MoE was not targeted. A full run from this state
would spend 1-4 h producing a checkpoint with its experts still in BF16.

It then crashed separately during calibration:

```
AttributeError: 'ZayaRotaryEmbedding' object has no attribute 'None_inv_freq'
```

### Correction: `register_zaya_moe.py` does not fix this pipeline

The registration is correct about *what* is wrong and would work inside
llm-compressor's `oneshot` pipeline. **But this script does not use that pipeline.**
It calls `compressed_tensors.apply_quantization_config(model, config)` directly
(lines 1085, 1870, 1970) and never imports `llmcompressor`. llm-compressor's MoE
linearizer runs *inside its own pipeline*, so a registry entry has no effect here.
The registry entry is kept because it is needed by whichever route is taken, but on
its own it changes nothing.

### Two real blockers, in priority order

**1. The experts are invisible to the quantizer.** `ZayaExperts` holds
`gate_up_proj` / `down_proj` as `nn.Parameter`; `apply_quantization_config` walks
`nn.Linear` modules. Options:

- **(a) Linearize explicitly before applying the config** — import
  `llmcompressor.modeling.moe.linearize` / `linear_experts` and convert
  `ZayaExperts` into per-expert `nn.Linear` modules in this script, then apply the
  config as now. Most surgical; reuses tested upstream code; keeps the bespoke
  calibration path intact. **Recommended.**
- **(b) Port the whole script to llm-compressor `oneshot`** — the registry entry
  then works as designed, but it discards this script's bespoke W4A4
  `input_global_scale` calibration, which is where much of the project's value is.
- (c) Teach `apply_quantization_config` about batched parameters — upstream work,
  out of scope.

**2. `ZayaRotaryEmbedding` / transformers 5 rotary API.** `None_inv_freq` means a
`layer_type` of `None` is being used to look up a per-layer-type `inv_freq` buffer.
The refactored config *does* declare `layer_types`, so something in the load or
calibration path is not passing it through. Must be fixed before any calibration
forward pass runs, independently of blocker 1.

### Calibration data — fixed, with one gap

`scripts/build_calibration_data.py` was broken against the transformers 5
chat-template API (commit `59bc382`); rebuilt output is
`data/calibration/calibration_data.pt`, **826 samples x 1024 tokens**.

It reproduces the published recipe's per-source counts **exactly** — math500 151,
gsm8k 153, humaneval 38, mbpp 25, alpaca 153, writingprompts 153, glaive 153 —
with one exception: **TriviaQA now yields 0 texts** where the published run got 153.
826 = 979 - 153. Fix the TriviaQA loader before the real run if calibration fidelity
to the published recipe matters; it is not a blocker for re-testing the gate.

`--arc-mix` is **not** the published recipe (it is the Phase 2 ARC-aware 11-source
mix). The published one is the **default**. An earlier version of this file said
otherwise.

### What is still true

Steps 3-5 are unchanged and still correct once a loadable checkpoint exists. The
CUDA-graph sweep remains blocked behind that.

---

## STEP 1 NOW PASSES — 2026-09-03, later. Step 2 is unblocked.

Five separate transformers-5 / llm-compressor breaks had to be fixed first; all
are committed. Dry run (4 layers, 8 samples) completes with `rc=0`:

```
Linearized MoE experts: nn.Linear count 361 -> 2281
Restored plain Linear.forward on 2000 quantized Linears
layer 4/4 done (3.1s elapsed, 146 hooks fired)
Set input_global_scale on 200/200 Linears (54 repaired from uncalibrated)
Compressed 200 Linear modules in 1s (skipped 0, BF16-exempted 0)
Saved in 2s | Total: 1596 MB
```

**Corrected gate numbers.** The earlier "~1,300 vs ~80" figure was an estimate.
The real numbers on this base:

| Signal | Expected | Means |
|---|---|---|
| `nn.Linear count` after linearization | **361 -> 2281** | +1,920 = 40 layers x 16 experts x 3 |
| `quantized Linears` | **2000** | 1,920 experts + 40 `o_proj` + 40 `mlp.gate.down_proj` |
| per layer | **50** | 48 expert + `o_proj` + `gate.down_proj` |

If linearization silently fails you get **80**, and the run now aborts on the
Linear-count assertion rather than proceeding.

### What was fixed to get here

1. **Calibration data** - `apply_chat_template` returns a `BatchEncoding` on
   transformers 5; `list(tokens)` was yielding dict KEYS, so each text
   contributed 2 strings instead of ~300 ids (`59bc382`).
2. **Rotary** - `forward(x, position_ids, layer_type)`; `layer_type` defaults to
   None in 5.14, so omitting it failed as `'ZayaRotaryEmbedding' object has no
   attribute 'None_inv_freq'` (`f34185c`).
3. **Experts invisible** - `ZayaExperts` holds batched `nn.Parameter`, so
   `targets: ["Linear"]` saw only 80 modules. Fixed by convert-after-load
   `linearize_moe` (`cc44458`).
4. **Decoder layer signature** - `residual` is gone and the return is a 2-tuple;
   the old call passed the residual positionally into the
   `prev_router_hidden_states` slot *and* again by keyword.
5. **Offloaded experts** - `LinearExperts2D.from_experts_module` calls
   `offload_module` on every expert, so `.to()` / `.data =` / rebinding all
   silently fail. Must run the forward inside
   `align_modules(list(layer.modules()), execution_device=dev)`.

### Watch this on the full run

`54/54 uncalibrated modules repaired` - **27% of the calibrated modules saw no
activations at all** in the dry run. With top-1 routing over 16 experts and only
8 samples, most experts are never selected, so this is expected *here*. On the
full 826-sample run it should fall close to zero.

**It is the number to check before trusting the output.** A module whose
`input_global_scale` was "repaired" rather than measured has a fabricated
activation scale, and W4A4 is exactly where that shows up as silent quality
loss. If the full run still reports a large repaired count, the calibration set
is not exercising the experts and the result should not be published.

### Step 2, ready to run

```bash
cd ~/zaya1-nvfp4-w4a4
ZAYA_MODEL_ID=Zyphra/ZAYA1-8B ~/quant-env/bin/python scripts/quantize_zaya_ct_nvfp4.py \
    --scheme w4a4 --output-dir ~/models/zaya-refactored-w4a4
```

Dry run was ~40 s of compute after a ~4 min load. The full run is 40 layers and
826 samples rather than 4 and 8, so budget hours and run it detached under tmux.
Steps 3-5 (load check, CUDA-graph sweep, re-eval) are unchanged.
