#!/usr/bin/env python3
"""Register `zaya` with llm-compressor's MoE expert linearizer.

Why this is needed
------------------
The refactored `Zyphra/ZAYA1-8B` stores its experts as batched `nn.Parameter`
tensors, not per-expert `nn.Linear` modules:

    class ZayaExperts(nn.Module):                  # modeling_zaya.py:576
        self.gate_up_proj = nn.Parameter(          # [E, 2*I, H]
            torch.empty(num_experts, 2 * intermediate_dim, hidden_dim))
        self.down_proj = nn.Parameter(             # [E, H, I]
            torch.empty(num_experts, hidden_dim, intermediate_dim))
    ...
        nn.functional.linear(current_state, self.gate_up_proj[expert_idx])

A quantization recipe with `targets: ["Linear"]` therefore **does not see the
experts at all**. On this model that silently leaves the entire MoE — the bulk of
the parameters — in BF16, and produces a barely-compressed checkpoint with no
error raised. That is a quiet wrong answer, not a crash, which is the failure mode
this project keeps having to guard against.

llm-compressor already solves this generically: `llmcompressor/modeling/moe/`
linearizes batched experts into real `nn.Linear` modules for the duration of
quantization, driven by the `ARCH_TO_IMPORT_PATHS` registry keyed on
`config.model_type`. `zaya` is simply not in that registry.

Why no custom conversion class is needed
----------------------------------------
`ExpertMLPWithGate.copy_from_experts_module` (linear_experts.py:55-58) reads the
non-transposed layout as:

    gate = gate_up_proj[i, :I]      up = gate_up_proj[i, I:]      down = down_proj[i]

which is exactly Zaya's `[E, 2I, H]` / `[E, H, I]`. `is_transposed` defaults to
False (helpers.py:56); only Llama4 overrides it to True. So the stock generic path
applies and a registry entry is sufficient.

Usage
-----
Import this module (or call `register()`) **before** building the quantization
pipeline, in the same process:

    from scripts.register_zaya_moe import register
    register()

Verified by source inspection against transformers 5.15.1 and llmcompressor
0.13.0. Confirm with `quantize_zaya_ct_nvfp4.py --dry-run` before a full run —
the dry run should report the expert Linears among the calibrated modules.
"""
from __future__ import annotations

CONFIG_PATH = "transformers.models.zaya.configuration_zaya.ZayaConfig"
EXPERTS_PATH = "transformers.models.zaya.modeling_zaya.ZayaExperts"


def register(strict: bool = True) -> bool:
    """Add `zaya` to llm-compressor's MoE conversion registry. Idempotent.

    Returns True if the entry is present afterwards. With strict=True, raises if
    llm-compressor's layout has changed enough that the registry is missing —
    better a loud failure here than a silent BF16 MoE later.
    """
    try:
        from llmcompressor.modeling.moe import conversion_mappings as cm
    except ImportError as exc:  # pragma: no cover
        if strict:
            raise RuntimeError(
                "llm-compressor MoE linearizer not importable; without it the "
                "ZAYA experts will silently NOT be quantized"
            ) from exc
        return False

    registry = getattr(cm, "ARCH_TO_IMPORT_PATHS", None)
    if registry is None:
        if strict:
            raise RuntimeError(
                "llmcompressor.modeling.moe.conversion_mappings.ARCH_TO_IMPORT_PATHS "
                "is gone - the registry contract changed, re-verify before quantizing"
            )
        return False

    registry.setdefault("zaya", (CONFIG_PATH, EXPERTS_PATH))
    return "zaya" in registry


def verify_expert_layout() -> None:
    """Fail loudly if ZayaExperts no longer matches the non-transposed contract."""
    import torch
    from transformers.models.zaya.modeling_zaya import ZayaExperts  # noqa: F401

    import inspect

    src = inspect.getsource(ZayaExperts.__init__)
    assert "gate_up_proj" in src and "down_proj" in src, (
        "ZayaExperts no longer exposes gate_up_proj/down_proj - the generic "
        "ExpertMLPWithGate path may not apply"
    )
    assert "2 * self.intermediate_dim" in src, (
        "gate_up_proj is no longer [E, 2*I, H]; check is_transposed and the "
        "gate/up split in linear_experts.py before trusting this registration"
    )
    del torch


if __name__ == "__main__":
    ok = register()
    print(f"zaya registered: {ok}")
    try:
        verify_expert_layout()
        print("expert layout matches the non-transposed ExpertMLPWithGate contract")
    except Exception as e:  # noqa: BLE001
        print(f"LAYOUT CHECK FAILED: {e}")
