# W4A4 NVFP4 vLLM source patches

This directory ships a single source-level patch against `vllm-src` that lets
ZAYA1-8B W4A4 NVFP4 checkpoints (e.g. `zaya1-8b-nvfp4-w4a4/`) load and
serve coherent text through vLLM's `VLLM_CUTLASS` MoE backend on consumer
Blackwell (RTX 5070 Ti, SM120).

This is distinct from `apply_zaya_patches.py`, which is a runtime
monkey-patcher for the **HuggingFace transformers** integration. This patch
applies to the **vLLM** source tree.

## What's in the patch

`zaya_w4a4_vllm_src.patch` modifies a single file:

- `vllm/model_executor/models/zaya.py` — three coupled fixes:
  1. `ZayaRouter.__init__` now threads `prefix=` through to the four
     `ReplicatedLinear` sub-modules. Without this, the router has empty
     default prefix and `CompressedTensorsConfig.ignore`'s
     `re:.*router.*` regex cannot match, so the router gets
     W4A4-quantized at model construction even though the checkpoint
     stores it as plain BF16. Result: ~160 weights silently skipped.
  2. `load_weights` routes per-Linear scalar `input_global_scale` and
     `weight_global_scale` to FusedMoE's `w13_/w2_{input,weight}_
     global_scale` params, replicating the same scalar across `w1`
     and `w3` slots for `linear_fc1`. Old behaviour fell through to
     the w13_weight split branch and hit `IndexError` on a 0-d tensor.
  3. `load_weights` splits `linear_fc1.weight_scale` into gate
     (`[:N]`) and up (`[N:]`) halves before loading. Loading once
     with `shard_id="w1"` only fills the gate half of FusedMoE's
     `w13_weight_scale`; the up half stays at `torch.empty()` NaN.

## Required base

The patch was generated against vLLM `v0.20.2` editable install with
Zyphra's Zaya overlay applied (registration in `models/registry.py`,
`cca.py`, `configs/zaya.py`, etc.). It does **not** install the overlay
itself — apply your overlay first, then this patch.

## How to apply

```bash
cd /path/to/vllm-src
patch -p1 < /path/to/zaya1-nvfp4-w4a4/patches/zaya_w4a4_vllm_src.patch
```

Dry-run first to verify it'll apply cleanly:

```bash
patch --dry-run -p1 < /path/to/zaya1-nvfp4-w4a4/patches/zaya_w4a4_vllm_src.patch
```

If the patch reports "Reversed (or previously applied) patch detected",
your tree already has the changes — you're done.

## Verifying success

After applying, run the smoke test against a W4A4 checkpoint:

```bash
python3 /path/to/zaya1-nvfp4-w4a4/scripts/test_zaya1_w4a4_inference.py
```

Expected output (the model continues differently across seeds; key signal
is non-pad, non-NaN tokens):

```
Prompt: "The capital of France is"
Output: " located in the north of the country. It is a city with a
         reputation of being very cosmopolitan and modern. ..."
Finish reason: length
```

A failure mode of `token_ids = [0] * N` and `logprob=nan` for the top
candidates indicates the patch was not applied or the checkpoint is missing
the `quantization_config` block / proper `weight_global_scale` values
(see `scripts/fix_w4a4_global_scales.py`).
