# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch
from torch.nn.parameter import Parameter
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

logger = init_logger(__name__)

__all__ = ["CompressedTensorsW4A16Fp4"]


def _dequant_fp4_weight(weight_packed: torch.Tensor) -> torch.Tensor:
    """Dequantize FP4 packed uint8 weights to bfloat16."""
    # weight_packed is uint8, shape [out, in // 2] with 2x FP4 values per byte
    # Extract low and high nibbles
    weight_low = weight_packed & 0x0F
    weight_high = (weight_packed >> 4) & 0x0F

    # FP4 E2M1 format: 1 sign, 2 exponent, 1 mantissa
    # Convert to float: (-1)^sign * 2^(exp-2) * (1 + mant/2) for normal values
    sign_low = (weight_low >> 3) & 0x1
    exp_low = (weight_low >> 1) & 0x3
    mant_low = weight_low & 0x1

    sign_high = (weight_high >> 3) & 0x1
    exp_high = (weight_high >> 1) & 0x3
    mant_high = weight_high & 0x1

    weight_float_low = ((-1.0) ** sign_low.float()) * (2.0 ** (exp_low.float() - 2.0)) * (1.0 + mant_low.float() / 2.0)
    weight_float_high = (
        ((-1.0) ** sign_high.float()) * (2.0 ** (exp_high.float() - 2.0)) * (1.0 + mant_high.float() / 2.0)
    )

    # Special case: exp=0, mant=0 → subnormal (0)
    # For simplicity, treat as 0 when both are zero
    is_zero_low = weight_low == 0
    is_zero_high = weight_high == 0
    weight_float_low = torch.where(is_zero_low, torch.zeros_like(weight_float_low), weight_float_low)
    weight_float_high = torch.where(is_zero_high, torch.zeros_like(weight_float_high), weight_float_high)

    # Interleave low and high to get [out, in] shape
    out_dim, in_half = weight_packed.shape
    weight_dequant = torch.empty(out_dim, in_half * 2, dtype=weight_float_low.dtype, device=weight_packed.device)
    weight_dequant[:, 0::2] = weight_float_low
    weight_dequant[:, 1::2] = weight_float_high

    return weight_dequant


class CompressedTensorsW4A16Fp4(CompressedTensorsScheme):
    def __init__(self):
        self.group_size = 16
        self._use_python_dequant = False

    @classmethod
    def get_min_capability(cls) -> int:
        # don't restrict as emulations
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # Global Weight Scale
        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        # Per Group Weight Scale
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        output_size = sum(layer.logical_widths)
        input_size = layer.input_size_per_partition

        # Check if Marlin can handle this layer
        # Marlin requires: size_k % 128 == 0, size_n % 64 == 0
        can_use_marlin = input_size % 128 == 0 and output_size % 64 == 0

        if not can_use_marlin:
            logger.info_once(
                "Skipping Marlin repack for layer (size_k=%d, size_n=%d). Using Python FP4 dequant fallback.",
                input_size,
                output_size,
            )
            self._use_python_dequant = True
            # Keep weight_packed as-is, no Marlin processing
            return

        self._use_python_dequant = False
        # Rename weight_packed to weight that marlin expects
        layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)
        del layer.weight_packed
        # ct stores the inverse of what is expected by the marlin kernel
        layer.weight_global_scale = Parameter(
            1.0 / layer.weight_global_scale.max().to(torch.float32), requires_grad=False
        )

        prepare_fp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._use_python_dequant:
            return self._apply_python_dequant(layer, x, bias)
        else:
            return apply_fp4_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                weight_global_scale=layer.weight_global_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )

    def _apply_python_dequant(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Python dequant fallback for layers where Marlin is incompatible."""
        # Dequantize weights from FP4 packed to bfloat16
        weight_packed = layer.weight_packed
        weight = _dequant_fp4_weight(weight_packed).to(x.dtype)

        # Apply per-group scales
        weight_scale = layer.weight_scale.to(x.dtype)
        weight_global_scale = 1.0 / layer.weight_global_scale.max().to(torch.float32)

        output_size = layer.output_size_per_partition
        input_size = layer.input_size_per_partition
        group_size = self.group_size

        # Reshape scales for broadcasting: [out, in/group] -> [out, in/group, 1] -> [out, in]
        weight_scale_expanded = weight_scale.unsqueeze(-1).repeat(1, 1, group_size).reshape(output_size, input_size)
        weight = weight * weight_scale_expanded * weight_global_scale

        # Compute linear
        out = torch.nn.functional.linear(x, weight, bias)
        return out
