#!/usr/bin/env python3
"""Fix: enable coherent NVFP4 CT text generation for ZAYA1-8B (May 2026).

After wsl_fix_moe_scale_routing.py + wsl_fix_marlin_group_size.py made the
model LOAD, three further bugs silently destroyed generation quality:

  1. ``FusedMoE._load_w13`` narrows the loaded tensor by half when
     ``is_act_and_mul=True``. Passing the combined [2*N, K//2] checkpoint with
     shard_id="w1" only loads the gate half; up rows stay uninitialized
     (``torch.empty`` garbage → NaN cascades on later layers). The fix splits
     both ``linear_fc1.weight_packed`` and ``linear_fc1.weight_scale`` into
     ``[:N]`` / ``[N:]`` and calls weight_loader twice as ``w1`` / ``w3``.
     (The "combined w13_weight_scale" fast path in fused_moe/layer.py:1298 only
     fires inside ``if "ModelOpt" in quant_method_name``, so CompressedTensors
     bypasses it and falls into the narrowing path.)

  2. The checkpoint stores ``lm_head.weight_packed`` / ``lm_head.weight_scale``
     (NVFP4), but ``ZayaForCausalLM`` creates ``ParallelLMHead(quant_config=None)``
     and registers an unquantized ``lm_head.weight``. With
     ``tie_word_embeddings=True``, the shared parameter is registered as
     ``model.embed_tokens.weight``. Both checkpoint keys were silently skipped
     by the default loader (and the broken f-string log line hid which keys).
     The fix buffers ``weight_packed`` / ``weight_scale`` during the load loop,
     dequantizes via ``compressed_tensors.unpack_fp4_from_uint8`` + ``dequantize``,
     and copies into ``model.embed_tokens.weight``.

  3. Rewrites ``CompressedTensorsW4A4Nvfp4MoEMethod`` to **Path A**: keep packed
     FP4 weights and per-group scales, dequant on the fly inside ``apply()``.
     Bypasses the broken Marlin MoE path (``nvfp4_marlin_process_scales``
     corrupts scales for this checkpoint) and the device-mismatched emulation
     backend. SwiGLU uses vLLM's standard ``silu(first_half) * second_half``
     convention; per-expert dispatch uses a manual loop over experts because
     ``fused_experts`` autotuner-on-Blackwell-without-cached-config produces
     zero output for constant-routed warmup batches.

Also fixes the broken log line (``"WARNING: key {chkpt_weight_name} ..."`` — a
literal-brace f-string without the ``f`` prefix that hides the actual key).

Inference contract after the fix:

  * ``dtype="bfloat16"`` is required. fp16 collapses output to a repeated
    token even with otherwise-correct weights — accumulation precision in the
    Python MoE dequant path is insufficient.
  * Expected greedy completion of "The capital of France is" → " Paris...".

The script is idempotent. Re-running after a vllm reinstall is safe; if an
expected anchor is missing it prints "[WARN] Pattern not found" and skips.
"""

from __future__ import annotations

from pathlib import Path

VLLM = Path("/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm")
ZAYA = VLLM / "model_executor/models/zaya.py"
MOE = (
    VLLM
    / "model_executor/layers/quantization/compressed_tensors/"
    / "compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py"
)


# ---------------------------------------------------------------------------
# Fix 1+2: zaya.py — split w13 (packed + scale) on load, dequant lm_head, log fmt
# ---------------------------------------------------------------------------
ZAYA_OLD_SCALE_LOAD = """                    if "weight_scale" in chkpt_weight_name:
                        param_name = f"{fused_moe_prefix}.w13_weight_scale"
                        param = params_dict.get(param_name)
                        if param is None:
                            param_name = param_name.replace(".zaya_block.", ".mlp.zaya_block.")
                            param = params_dict.get(param_name)
                        if param is None:
                            raise KeyError(f"FusedMoE w13_weight_scale not found for {fused_moe_prefix}")
                        fused_moe_module.weight_loader(
                            param, loaded_weight, chkpt_weight_name, "w1", expert_id
                        )
                        loaded_params.add(param_name)
                        continue"""

ZAYA_NEW_SCALE_LOAD = """                    if "weight_scale" in chkpt_weight_name:
                        param_name = f"{fused_moe_prefix}.w13_weight_scale"
                        param = params_dict.get(param_name)
                        if param is None:
                            param_name = param_name.replace(".zaya_block.", ".mlp.zaya_block.")
                            param = params_dict.get(param_name)
                        if param is None:
                            raise KeyError(f"FusedMoE w13_weight_scale not found for {fused_moe_prefix}")
                        # Same gate/up split as weight_packed: the combined
                        # w13_weight_scale path in FusedMoE.weight_loader is
                        # gated on `"ModelOpt" in quant_method_name`, so for
                        # CompressedTensors the scale falls into _load_w13
                        # which only loads gate-half. Split here.
                        half = loaded_weight.shape[0] // 2
                        gate_scale = loaded_weight[:half, :]
                        up_scale = loaded_weight[half:, :]
                        fused_moe_module.weight_loader(
                            param, gate_scale, chkpt_weight_name, "w1", expert_id
                        )
                        fused_moe_module.weight_loader(
                            param, up_scale, chkpt_weight_name, "w3", expert_id
                        )
                        loaded_params.add(param_name)
                        continue"""

ZAYA_OLD_W13_LOAD = """                    if "_packed" in param_name:
                        fused_moe_module.weight_loader(
                            param, loaded_weight, chkpt_weight_name, "w1", expert_id
                        )
                    else:
                        half = loaded_weight.shape[0] // 2
                        gate_weight = loaded_weight[:half, :]
                        up_weight = loaded_weight[half:, :]
                        fused_moe_module.weight_loader(
                            param, gate_weight, chkpt_weight_name, "w1", expert_id
                        )
                        fused_moe_module.weight_loader(
                            param, up_weight, chkpt_weight_name, "w3", expert_id
                        )"""

ZAYA_NEW_W13_LOAD = """                    # FusedMoE.weight_loader with shard_id="w1" narrows the
                    # loaded tensor to gate-half (first N rows) — passing the
                    # combined [2*N, K//2] packed checkpoint with shard_id="w1"
                    # only loads gate; up rows stay uninitialized. Split here
                    # for both packed and non-packed.
                    half = loaded_weight.shape[0] // 2
                    gate_weight = loaded_weight[:half, :]
                    up_weight = loaded_weight[half:, :]
                    fused_moe_module.weight_loader(
                        param, gate_weight, chkpt_weight_name, "w1", expert_id
                    )
                    fused_moe_module.weight_loader(
                        param, up_weight, chkpt_weight_name, "w3", expert_id
                    )"""

ZAYA_OLD_LOAD_OTHER = """        loaded_params: set[str] = set()
        import re

        import tqdm"""

ZAYA_NEW_LOAD_OTHER = """        loaded_params: set[str] = set()
        lm_head_buf: dict[str, torch.Tensor] = {}
        import re

        import tqdm"""

ZAYA_OLD_FALLTHROUGH = """            # Loading other parameters
            if chkpt_weight_name not in params_dict:
                logger.info(
                    "WARNING: key {chkpt_weight_name} not in params! Skipping loading"
                )
                continue
            param = params_dict.get(chkpt_weight_name)
            if param is None:
                alt = chkpt_weight_name.replace(".zaya_block.", ".mlp.zaya_block.")
                param = params_dict.get(alt)
            if param is None:
                raise KeyError(f"{chkpt_weight_name}")
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(chkpt_weight_name)
        return loaded_params"""

ZAYA_NEW_FALLTHROUGH = """            # lm_head is created as an unquantized ParallelLMHead (the
            # _FP32EmbeddingMethod path reads layer.weight directly), but
            # the checkpoint stores it as NVFP4: lm_head.weight_packed +
            # lm_head.weight_scale. Buffer here, dequant after the loop.
            if chkpt_weight_name == "lm_head.weight_packed":
                lm_head_buf["packed"] = loaded_weight
                continue
            if chkpt_weight_name == "lm_head.weight_scale":
                lm_head_buf["scale"] = loaded_weight
                continue

            # Loading other parameters
            if chkpt_weight_name not in params_dict:
                logger.info(
                    "WARNING: key %s not in params! Skipping loading",
                    chkpt_weight_name,
                )
                continue
            param = params_dict.get(chkpt_weight_name)
            if param is None:
                alt = chkpt_weight_name.replace(".zaya_block.", ".mlp.zaya_block.")
                param = params_dict.get(alt)
            if param is None:
                raise KeyError(f"{chkpt_weight_name}")
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(chkpt_weight_name)

        # Finalize lm_head after the loop: dequant NVFP4 → fp16 into lm_head.weight.
        if "packed" in lm_head_buf and "scale" in lm_head_buf:
            from compressed_tensors.compressors.nvfp4.helpers import (
                unpack_fp4_from_uint8,
            )
            from compressed_tensors.quantization.lifecycle.forward import (
                dequantize,
            )
            packed = lm_head_buf["packed"]
            scale = lm_head_buf["scale"]
            m, nh = packed.shape
            wq = unpack_fp4_from_uint8(packed, m, nh * 2, dtype=torch.float32)
            w = dequantize(x_q=wq, scale=scale.float(), dtype=torch.float32)
            # With tie_word_embeddings=True the parameter is registered as
            # model.embed_tokens.weight; lm_head shares the same tensor.
            target = params_dict.get("lm_head.weight") or params_dict.get(
                "model.embed_tokens.weight"
            )
            if target is None:
                logger.warning(
                    "lm_head/embed_tokens param not found; cannot bind NVFP4 lm_head"
                )
            else:
                w = w.to(target.dtype).to(target.device)
                with torch.no_grad():
                    if w.shape == target.shape:
                        target.data.copy_(w)
                    else:
                        mn = min(w.shape[0], target.shape[0])
                        target.data[:mn].copy_(w[:mn])
                        if mn < target.shape[0]:
                            target.data[mn:].zero_()
                loaded_params.add("model.embed_tokens.weight")
        return loaded_params"""


def fix_zaya() -> bool:
    if not ZAYA.exists():
        print(f"  [ERROR] {ZAYA} not found")
        return False
    src = ZAYA.read_text()
    new = src

    if ZAYA_OLD_SCALE_LOAD in new:
        new = new.replace(ZAYA_OLD_SCALE_LOAD, ZAYA_NEW_SCALE_LOAD)
        print("  [FIX] Split w13_weight_scale into gate/up halves on load")
    elif "Same gate/up split as weight_packed" in new:
        print("  [OK]  w13_weight_scale already split")
    else:
        print("  [WARN] Could not find w13_weight_scale load block")

    if ZAYA_OLD_W13_LOAD in new:
        new = new.replace(ZAYA_OLD_W13_LOAD, ZAYA_NEW_W13_LOAD)
        print("  [FIX] Split w13_weight_packed into gate/up halves on load")
    elif 'FusedMoE.weight_loader with shard_id="w1" narrows' in new or 'shard_id="w1" narrows' in new:
        print("  [OK]  w13_weight_packed already split")
    else:
        print("  [WARN] Could not find w13 packed load block")

    if ZAYA_OLD_LOAD_OTHER in new:
        new = new.replace(ZAYA_OLD_LOAD_OTHER, ZAYA_NEW_LOAD_OTHER)
        print("  [FIX] Declared lm_head_buf for tied embedding dequant")
    elif "lm_head_buf: dict[str, torch.Tensor]" in new:
        print("  [OK]  lm_head_buf already declared")
    else:
        print("  [WARN] Could not find loaded_params declaration")

    if ZAYA_OLD_FALLTHROUGH in new:
        new = new.replace(ZAYA_OLD_FALLTHROUGH, ZAYA_NEW_FALLTHROUGH)
        print("  [FIX] Added lm_head dequant + fixed broken log f-string")
    elif "Finalize lm_head after the loop" in new:
        print("  [OK]  lm_head dequant already wired")
    else:
        print("  [WARN] Could not find load-other / lm_head fallthrough block")

    if new == src:
        return False
    ZAYA.write_text(new)
    print(f"  [SAVED] {ZAYA.name}")
    return True


# ---------------------------------------------------------------------------
# Fix 3: rewrite MoE method for Path A on-the-fly Python dequant.
# ---------------------------------------------------------------------------
MOE_TARGET = '''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    is_global_sf_supported_for_nvfp4_backend,
    select_nvfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class CompressedTensorsW4A4Nvfp4MoEMethod(CompressedTensorsMoEMethod):
    def __init__(
        self,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
        use_a16: bool = False,
        group_size: int = 16,
    ):
        super().__init__(moe)
        self.group_size = group_size

        # Select experts implementation.
        self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
            config=self.moe,
            weight_key=kNvfp4Static,
            activation_key=None if use_a16 else kNvfp4Dynamic,
        )

        self.use_global_sf = is_global_sf_supported_for_nvfp4_backend(
            self.nvfp4_backend
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Weight Scales
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Weight Global Scales (this checkpoint has no `weight_scale_2` keys;
        # initialize to ones so the dequant fallback is a no-op rescale).
        w13_weight_scale_2 = torch.nn.Parameter(
            torch.ones(num_experts, w13_num_shards, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_global_scale", w13_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale_2, extra_weight_attrs)

        w2_weight_scale_2 = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_weight_global_scale", w2_weight_scale_2)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_weight_scale_2, extra_weight_attrs)

        # Input Global Scales (unused on the W4A16 path; left at ones).
        w13_input_scale = torch.nn.Parameter(
            torch.ones(num_experts, w13_num_shards, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_input_global_scale", w13_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_input_global_scale", w2_input_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        """Path A: keep packed FP4 weights and per-group scales, dequant
        on the fly in apply(). Marlin kernel bypassed entirely because
        nvfp4_marlin_process_scales corrupts MoE scales for this checkpoint.
        Clone so downstream Marlin-prep passes can't mutate in place.
        """
        layer.w13_weight = layer.w13_weight_packed.data.clone()
        delattr(layer, "w13_weight_packed")
        layer.w2_weight = layer.w2_weight_packed.data.clone()
        delattr(layer, "w2_weight_packed")
        self._w13_scale = layer.w13_weight_scale.data.clone()
        self._w2_scale = layer.w2_weight_scale.data.clone()

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        from vllm.model_executor.layers.fused_moe.config import (
            FUSED_MOE_UNQUANTIZED_CONFIG,
        )
        return FUSED_MOE_UNQUANTIZED_CONFIG

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Path A: on-the-fly dequant + manual per-expert SwiGLU MoE.

        Bypasses `fused_experts` because the Triton autotuner produces zero
        output on constant-routed warmup batches without a cached config for
        sm_120. The manual loop is slower but correct.
        """
        # Dequantize all experts to x.dtype.
        w13_fp = self._dequant_experts(
            layer.w13_weight, self._w13_scale, out_dtype=x.dtype
        )
        w2_fp = self._dequant_experts(
            layer.w2_weight, self._w2_scale, out_dtype=x.dtype
        )

        # Manual MoE: for each token, route to top-k experts.
        # SwiGLU: silu(gate) * up where w13 = [gate; up] along the first half.
        E = w13_fp.shape[0]
        T, K = x.shape
        out = torch.zeros(T, K, dtype=x.dtype, device=x.device)

        for e_id in range(E):
            mask = topk_ids == e_id  # [T, topk] bool
            if not mask.any():
                continue
            # Tokens whose top-k routes include expert e
            token_idx = mask.any(dim=-1).nonzero(as_tuple=True)[0]
            if token_idx.numel() == 0:
                continue
            xe = x[token_idx]  # [t_e, K]
            # gate+up = xe @ w13[e].T → [t_e, 2N]
            gate_up = torch.nn.functional.linear(xe, w13_fp[e_id])
            N = gate_up.shape[-1] // 2
            gate, up = gate_up[..., :N], gate_up[..., N:]
            # vLLM SiluAndMul convention: silu(first_half) * second_half.
            hidden = torch.nn.functional.silu(gate) * up
            # down = hidden @ w2[e].T → [t_e, K]
            down = torch.nn.functional.linear(hidden, w2_fp[e_id])
            # Apply per-token expert weight. A token may route to expert e
            # in any of its top-k slots; sum contributions over slots.
            tw = (
                (topk_ids[token_idx] == e_id).to(x.dtype)
                * topk_weights[token_idx].to(x.dtype)
            ).sum(dim=-1, keepdim=True)
            out[token_idx] = out[token_idx] + tw * down

        return out

    @staticmethod
    def _dequant_experts(
        packed: torch.Tensor,  # [E, M, P//2] uint8
        scales: torch.Tensor,  # [E, M, P//gs] fp8
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Dequantize all experts from FP4 packed to `out_dtype`. Mirrors the
        Linear NVFP4 Python-dequant fallback (which is known to produce correct
        attention output on CCA projections)."""
        from compressed_tensors.compressors.nvfp4.helpers import (
            unpack_fp4_from_uint8,
        )
        from compressed_tensors.quantization.lifecycle.forward import (
            dequantize,
        )
        E, M, Ph = packed.shape
        P = Ph * 2
        result = []
        for e in range(E):
            wq = unpack_fp4_from_uint8(packed[e], M, P)
            w = dequantize(
                x_q=wq, scale=scales[e].float(), dtype=scales[e].float().dtype
            )
            result.append(w.to(out_dtype))
        return torch.stack(result).contiguous()
'''


def fix_moe() -> bool:
    if not MOE.exists():
        print(f"  [ERROR] {MOE} not found")
        return False
    current = MOE.read_text()
    sentinel = "Path A: keep packed FP4 weights and per-group scales"
    if sentinel in current and "Path A: on-the-fly dequant + manual per-expert" in current:
        print("  [OK]  MoE Path A already installed")
        return False
    MOE.write_text(MOE_TARGET)
    print(f"  [FIX] Rewrote {MOE.name} for Path A on-the-fly Python dequant")
    return True


def main() -> int:
    print("Applying NVFP4 CT text-generation fixes (session 2, May 2026)")
    print()
    print("Fix 1+2: zaya.py — split w13 halves, dequant tied lm_head, fix log fmt")
    fix_zaya()
    print()
    print("Fix 3: compressed_tensors_moe_w4a4_nvfp4.py — Path A on-the-fly dequant")
    fix_moe()
    print()
    print("Done. Re-run smoke test with dtype='bfloat16' to verify coherent text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
