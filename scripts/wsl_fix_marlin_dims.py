#!/usr/bin/env python3
"""Fix: Properly handle CCA N=17 layers in W4A16 scheme.

Strategy: For layers with N % 64 != 0 (CCA latent dims), do the standard
weight_packed → weight rename AND compute weight_global_scale, but skip
Marlin repack. Clone weight data for Python dequant fallback.
"""

from pathlib import Path

P = Path(
    "/home/ttimm/vllm-src/vllm/model_executor/layers/quantization/"
    "compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py"
)

content = P.read_text()

# Restore from git first to get clean slate
import subprocess  # noqa: E402

repo = "/home/ttimm/vllm-src"
rel = "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py"
subprocess.run(["git", "-C", repo, "checkout", "--", rel], capture_output=True)
content = P.read_text()

# Now apply a clean, correct fix
lines = content.split("\n")
new_lines = []
i = 0

while i < len(lines):
    line = lines[i]

    # process_weights_after_loading: replace the simple call with guarded version
    if line.strip() == "def process_weights_after_loading(self, layer: torch.nn.Module) -> None:":
        new_lines.append(line)
        new_lines.append("        # Check dimension alignment for Marlin repack.")
        new_lines.append("        # Marlin requires K % 256 == 0, N % 64 == 0.")
        new_lines.append("        size_n = layer.output_size_per_partition")
        new_lines.append("        size_k = layer.input_size_per_partition")
        new_lines.append("        dims_ok = (size_n % 64 == 0) and (size_k % 256 == 0)")
        new_lines.append("        gs_ok = self.group_size == 16")
        new_lines.append("")
        new_lines.append("        if not dims_ok or not gs_ok:")
        new_lines.append("            import logging")
        new_lines.append("            _lg = logging.getLogger(__name__)")
        new_lines.append("            _lg.warning(")
        new_lines.append('                "Skipping Marlin repack: N=%d K=%d gs=%d.",')
        new_lines.append("                size_n, size_k, self.group_size,")
        new_lines.append("            )")
        new_lines.append("            # Do the standard weight rename (required for forward pass)")
        new_lines.append('            if hasattr(layer, "weight_packed"):')
        new_lines.append("                layer._weight_packed_data = layer.weight_packed.data.clone()")
        new_lines.append("                layer._weight_scale_data = layer.weight_scale.data.clone()")
        new_lines.append("                layer._weight_global_scale_data = layer.weight_global_scale.data.clone()")
        new_lines.append("                layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)")
        new_lines.append("                del layer.weight_packed")
        new_lines.append("                layer.weight_global_scale = Parameter(")
        new_lines.append("                    1.0 / layer.weight_global_scale.max().to(torch.float32),")
        new_lines.append("                    requires_grad=False,")
        new_lines.append("                )")
        new_lines.append("            layer._marlin_repack_skipped = True")
        new_lines.append("            return")
        new_lines.append("")
        new_lines.append("        # Process parameters for marlin repacking")
        new_lines.append("        layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)")
        new_lines.append("        del layer.weight_packed")
        new_lines.append("        layer.weight_global_scale = Parameter(")
        new_lines.append("            1.0 / layer.weight_global_scale.max().to(torch.float32), requires_grad=False")
        new_lines.append("        )")
        new_lines.append("        prepare_fp4_layer_for_marlin(layer)")
        # Skip original method body
        i += 1
        while i < len(lines):
            if lines[i].startswith("    def ") and i > 1:
                break
            i += 1
        continue

    # apply_weights: add fallback check
    if line.strip() == "def apply_weights(":
        new_lines.append(line)
        new_lines.append("        self,")
        new_lines.append("        layer: torch.nn.Module,")
        new_lines.append("        x: torch.Tensor,")
        new_lines.append("        bias: torch.Tensor | None = None,")
        new_lines.append("    ) -> torch.Tensor:")
        new_lines.append('        if getattr(layer, "_marlin_repack_skipped", False):')
        new_lines.append('            if hasattr(layer, "_weight_packed_data"):')
        new_lines.append(
            "                from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8"
        )
        new_lines.append("                from compressed_tensors.quantization.lifecycle.forward import dequantize")
        new_lines.append("                wq = layer._weight_packed_data.to(x.device)")
        new_lines.append("                ws = layer._weight_scale_data.to(x.device)")
        new_lines.append("                wgs = layer._weight_global_scale_data.to(x.device)")
        new_lines.append("                m = layer.output_size_per_partition")
        new_lines.append("                nh = layer.input_size_per_partition // 2")
        new_lines.append("                w = unpack_fp4_from_uint8(wq, m, nh * 2)")
        new_lines.append(
            "                w = dequantize(x_q=w, scale=ws.float(), global_scale=wgs, dtype=ws.float().dtype)"
        )
        new_lines.append("                out = torch.nn.functional.linear(x, w.to(x.dtype))")
        new_lines.append("                return out + bias if bias is not None else out")
        new_lines.append("            # Fallback for wrapper modules: use sub-module or skip")
        new_lines.append("            return apply_fp4_marlin_linear(")
        new_lines.append("                input=x, weight=layer.weight, weight_scale=layer.weight_scale,")
        new_lines.append("                weight_global_scale=layer.weight_global_scale, workspace=layer.workspace,")
        new_lines.append(
            "                size_n=layer.output_size_per_partition, size_k=layer.input_size_per_partition,"
        )
        new_lines.append("                bias=bias,")
        new_lines.append("            )")
        new_lines.append("        return apply_fp4_marlin_linear(")
        new_lines.append("            input=x, weight=layer.weight, weight_scale=layer.weight_scale,")
        new_lines.append("            weight_global_scale=layer.weight_global_scale, workspace=layer.workspace,")
        new_lines.append("            size_n=layer.output_size_per_partition, size_k=layer.input_size_per_partition,")
        new_lines.append("            bias=bias,")
        new_lines.append("        )")
        # Skip original method body
        i += 1
        while i < len(lines):
            if lines[i].startswith("    def ") or lines[i].startswith("class "):
                break
            i += 1
        continue

    new_lines.append(line)
    i += 1

P.write_text("\n".join(new_lines))
print("Done.")
