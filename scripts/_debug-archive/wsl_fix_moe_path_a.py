#!/usr/bin/env python3
"""Apply Path A MoE fix to vllm-src: bypass Marlin, use on-the-fly Python dequant.

Session 2 fix adapted for vllm-src editable install at /home/ttimm/vllm-src/.
Bypasses Marlin MoE kernel (which corrupts FP8_E4M3 scales via S0E5M3 sign flip)
and uses correct Python dequant identical to the working Linear fallback path.
"""

from pathlib import Path

MOE = Path(
    "/home/ttimm/vllm-src/vllm/model_executor/layers/quantization/"
    "compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py"
)

content = MOE.read_text()

# Check if already patched
if "Path A: keep packed FP4 weights" in content:
    print("Already patched — Path A MoE dequant present.")
    exit(0)

# ── Fix 1: process_weights_after_loading ──
old_proc = '''    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        """
        Convert NVFP4 MoE weights into kernel format and setup the kernel.
        """
        # NOTE(rob): wN_weight_packed -> wN_weight is because ModularKernelMethod
        # requires this naming convention. However, the name change breaks
        # reloading because the state dict no longer matches disk. Once we
        # remove MKM, we should revert this change to ensure compatibility.
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        # Use a single gscale for w13.
        if self.moe.is_act_and_mul and not torch.allclose(
            layer.w13_weight_global_scale[:, 0], layer.w13_weight_global_scale[:, 1]
        ):
            logger.warning_once(
                "w1_weight_global_scale must match w3_weight_global_scale. "
                "Accuracy may be affected.",
            )
        w13_weight_global_scale = layer.w13_weight_global_scale[:, 0].contiguous()

        # Shuffle weights into the NvFp4 kernel format.
        (
            w13,
            w13_scale,
            w13_scale_2,
            a13_scale,
            w2,
            w2_scale,
            w2_scale_2,
            a2_scale,
        ) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=self.nvfp4_backend,
            layer=layer,
            w13=layer.w13_weight,
            w13_scale=layer.w13_weight_scale,
            w13_scale_2=(1.0 / w13_weight_global_scale),
            a13_scale=(1.0 / layer.w13_input_global_scale),
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            w2_scale_2=(1.0 / layer.w2_weight_global_scale),
            a2_scale=(1.0 / layer.w2_input_global_scale),
            is_act_and_mul=self.moe.is_act_and_mul,
        )

        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w2_weight_scale", w2_scale)
        layer.w13_weight_scale_2 = w13_scale_2
        layer.w2_weight_scale_2 = w2_scale_2
        layer.w13_input_scale = a13_scale
        layer.w2_input_scale = a2_scale

        # Setup modular kernel.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.experts_cls is not None
        self.moe_kernel = make_nvfp4_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            shared_experts=layer.shared_experts,
            routing_tables=layer._maybe_init_expert_routing_tables(),
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)'''

new_proc = '''    def process_weights_after_loading(self, layer: FusedMoE) -> None:
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
        self._w2_scale = layer.w2_weight_scale.data.clone()'''

if old_proc in content:
    content = content.replace(old_proc, new_proc)
    print("  [FIX 1] process_weights_after_loading → Path A clone-only")
else:
    print("  [SKIP 1] process_weights_after_loading not found (may already be patched)")

# ── Fix 2: get_fused_moe_quant_config ──
old_qc = """    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        return make_nvfp4_moe_quant_config(
            backend=self.nvfp4_backend,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_scale_2=layer.w13_weight_scale_2,
            w2_scale_2=layer.w2_weight_scale_2,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )"""

new_qc = """    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        from vllm.model_executor.layers.fused_moe.config import (
            FUSED_MOE_UNQUANTIZED_CONFIG,
        )
        return FUSED_MOE_UNQUANTIZED_CONFIG"""

if old_qc in content:
    content = content.replace(old_qc, new_qc)
    print("  [FIX 2] get_fused_moe_quant_config → UNQUANTIZED")
else:
    print("  [SKIP 2] get_fused_moe_quant_config not found")

# ── Fix 3: apply ──
old_apply = """    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts_input=shared_experts_input,
        )"""

new_apply = '''    def apply(
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
        w13_fp = self._dequant_experts(
            layer.w13_weight, self._w13_scale, out_dtype=x.dtype
        )
        w2_fp = self._dequant_experts(
            layer.w2_weight, self._w2_scale, out_dtype=x.dtype
        )

        E = w13_fp.shape[0]
        T, K = x.shape
        out = torch.zeros(T, K, dtype=x.dtype, device=x.device)

        for e_id in range(E):
            mask = topk_ids == e_id
            if not mask.any():
                continue
            token_idx = mask.any(dim=-1).nonzero(as_tuple=True)[0]
            if token_idx.numel() == 0:
                continue
            xe = x[token_idx]
            gate_up = torch.nn.functional.linear(xe, w13_fp[e_id])
            N = gate_up.shape[-1] // 2
            gate, up = gate_up[..., :N], gate_up[..., N:]
            hidden = torch.nn.functional.silu(gate) * up
            down = torch.nn.functional.linear(hidden, w2_fp[e_id])
            tw = (
                (topk_ids[token_idx] == e_id).to(x.dtype)
                * topk_weights[token_idx].to(x.dtype)
            ).sum(dim=-1, keepdim=True)
            out[token_idx] = out[token_idx] + tw * down

        return out

    @staticmethod
    def _dequant_experts(
        packed: torch.Tensor,
        scales: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
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
        return torch.stack(result).contiguous()'''

if old_apply in content:
    content = content.replace(old_apply, new_apply)
    print("  [FIX 3] apply → Path A Python dequant MoE")
else:
    print("  [SKIP 3] apply not found")

# ── Also remove maybe_make_prepare_finalize if it exists (not needed for Path A) ──
# We keep it since it just raises ValueError (it's a stub)

MOE.write_text(content)
print("\nDone. Path A MoE fix applied to vllm-src.")
