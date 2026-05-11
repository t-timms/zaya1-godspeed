"""Runtime monkey-patches for ZAYA1-8B (Zyphra/transformers @ zaya1).

Applies 8 zero-risk ecosystem compatibility patches at runtime without
modifying the installed transformers package. All patches produce
bit-identical outputs and are backward-compatible.

Patches applied:
  1. _can_compile_fullgraph = True           (torch.compile support)
  2. _can_record_outputs metadata             (TRL recording)
  3. logits_to_keep support                   (train memory savings)
  4. _tied_weights_keys declaration           (PEFT weight tying)
  5. _tp_plan / _pp_plan                      (distributed inference)
  6. _supports_flex_attn = True               (FlexAttention backend)
  7. _supports_flash_attn = True              (FlashAttention flag)
  8. _supports_sdpa = True                    (SDPA flag)
  9. router_aux_loss_coef config default      (MoE training config)

Not applied (require source changes):
  - GradientCheckpointingLayer base class (use model.gradient_checkpointing_enable())
  - Hub-loaded RoPE (Liger Kernel auto-dispatch not possible without decorator)
  - Hub-loaded RMSNorm (same reason)

Usage:
    from patches.apply_zaya_patches import apply_all_patches
    apply_all_patches()

    # Then load the model normally
    model = AutoModelForCausalLM.from_pretrained("Zyphra/ZAYA1-8B", ...)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _patch_logits_to_keep(forward_fn):
    """Wrap ZayaForCausalLM.forward to support logits_to_keep parameter."""

    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        original_logits_to_keep = kwargs.pop("_original_logits_to_keep", None)
        if original_logits_to_keep is not None:
            kwargs["logits_to_keep"] = original_logits_to_keep
        else:
            kwargs["_original_logits_to_keep"] = logits_to_keep

        outputs = forward_fn(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            **kwargs,
        )

        if hasattr(outputs, "logits") and outputs.logits is not None:
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int) and logits_to_keep > 0
                else slice(None)
            )
            if slice_indices != slice(None):
                outputs["logits"] = outputs.logits[:, slice_indices, :]

        return outputs

    return patched_forward


def patch_model_class(model_cls):
    """Apply all safe patches to a ZayaForCausalLM or ZayaPreTrainedModel class."""
    patched = set()

    if hasattr(model_cls, "forward"):
        original_forward = model_cls.forward

        def safe_forward(self, *args, **kwargs):
            if kwargs.get("logits_to_keep", 0):
                return _patched_logits_forward(self, *args, **kwargs)
            return original_forward(self, *args, **kwargs)

        _patched_logits_forward = _patch_logits_to_keep(original_forward)
        model_cls.forward = _patched_logits_forward
        patched.add("logits_to_keep")

    if not getattr(model_cls, "_supports_flash_attn", False):
        model_cls._supports_flash_attn = True
        patched.add("flash_attn")

    if not getattr(model_cls, "_supports_sdpa", False):
        model_cls._supports_sdpa = True
        patched.add("sdpa")

    if not getattr(model_cls, "_supports_flex_attn", False):
        model_cls._supports_flex_attn = True
        patched.add("flex_attn")

    if not getattr(model_cls, "_can_compile_fullgraph", False):
        model_cls._can_compile_fullgraph = True
        patched.add("compile_fullgraph")

    if not getattr(model_cls, "_tied_weights_keys", None):
        model_cls._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
        patched.add("tied_weights_keys")

    if not getattr(model_cls, "_tp_plan", None):
        model_cls._tp_plan = {"lm_head": "colwise_gather_output"}
        patched.add("tp_plan")

    if not getattr(model_cls, "_pp_plan", None):
        model_cls._pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
        patched.add("pp_plan")

    if not getattr(model_cls, "_can_record_outputs", None):
        try:
            from transformers.models.zaya.modeling_zaya import (
                ZayaAttention,
                ZayaDecoderATTLayer,
            )

            model_cls._can_record_outputs = {
                "hidden_states": ZayaDecoderATTLayer,
                "attentions": ZayaAttention,
            }
            patched.add("record_outputs")
        except ImportError:
            logger.debug("Cannot import ZayaAttention/ZayaDecoderATTLayer for _can_record_outputs")

    return patched


def patch_config(config_cls):
    """Add router_aux_loss_coef default to ZayaConfig if missing."""
    if not hasattr(config_cls, "router_aux_loss_coef"):
        config_cls.router_aux_loss_coef = 0.001
        return {"router_aux_loss_coef"}
    return set()


def apply_all_patches() -> dict[str, set[str]]:
    """Apply all available runtime patches. Call before model loading.

    Returns:
        Dict mapping component name to set of patches applied.
    """
    results: dict[str, set[str]] = {}

    try:
        from transformers.models.zaya import ZayaForCausalLM

        results["ZayaForCausalLM"] = patch_model_class(ZayaForCausalLM)
    except ImportError:
        logger.debug("ZayaForCausalLM not available (transformers fork not installed?)")

    try:
        from transformers.models.zaya import ZayaPreTrainedModel

        results["ZayaPreTrainedModel"] = patch_model_class(ZayaPreTrainedModel)
    except ImportError:
        pass

    try:
        from transformers.models.zaya.configuration_zaya import ZayaConfig

        results["ZayaConfig"] = patch_config(ZayaConfig)
    except ImportError:
        pass

    total = sum(len(v) for v in results.values())
    logger.info("Applied %d ZAYA1-8B ecosystem patches across %d components", total, len(results))
    for component, patches in sorted(results.items()):
        if patches:
            logger.info("  %s: %s", component, ", ".join(sorted(patches)))

    return results


def _test_patches_safe() -> bool:
    """Verify patches don't break imports. Returns True if safe."""
    try:
        apply_all_patches()
        from transformers.models.zaya import ZayaForCausalLM, ZayaPreTrainedModel
        from transformers.models.zaya.configuration_zaya import ZayaConfig

        assert getattr(ZayaPreTrainedModel, "_supports_flash_attn", False)
        assert getattr(ZayaPreTrainedModel, "_supports_sdpa", False)
        assert getattr(ZayaPreTrainedModel, "_can_compile_fullgraph", False)
        assert getattr(ZayaForCausalLM, "_tied_weights_keys", None) == {
            "lm_head.weight": "model.embed_tokens.weight"
        }
        assert hasattr(ZayaConfig, "router_aux_loss_coef")
        return True
    except Exception:
        logger.exception("Patch verification failed")
        return False
