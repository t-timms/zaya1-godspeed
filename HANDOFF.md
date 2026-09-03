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

## Step 1 — dry-run the re-quantization (~10 min GPU) ← START HERE

```bash
cd ~/zaya1-nvfp4-w4a4
~/quant-env/bin/python -c "import sys; sys.path.insert(0,'scripts'); \
  from register_zaya_moe import register; register()"   # sanity, should print nothing
ZAYA_MODEL_ID=Zyphra/ZAYA1-8B ~/quant-env/bin/python scripts/quantize_zaya_ct_nvfp4.py \
    --scheme w4a4 --dry-run
```

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
