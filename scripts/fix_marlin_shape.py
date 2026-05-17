"""Fix CompressedTensorsW4A16Fp4 to handle incompatible Marlin shapes."""
path = "/home/ttimm/vllm-src/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py"
with open(path) as f:
    content = f.read()

# Replace the process_weights_after_loading to guard against incompatible shapes
old = '''    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Process parameters for marlin repacking

        # Rename weight_packed to weight that marlin expects
        layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)
        del layer.weight_packed
        # ct stores the inverse of what is expected by the marlin kernel
        layer.weight_global_scale = Parameter(
            1.0 / layer.weight_global_scale.max().to(torch.float32), requires_grad=False
        )

        prepare_fp4_layer_for_marlin(layer)'''

new = '''    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Process parameters for marlin repacking
        # Skip Marlin if shapes are incompatible (e.g., size_n not divisible by tile)
        output_size = sum(layer.logical_widths)
        input_size = layer.input_size_per_partition
        
        # Marlin requires size_k % 128 == 0 and size_n % 64 == 0
        if input_size % 128 != 0 or output_size % 64 != 0:
            logger.info(
                "Skipping Marlin repack: size_k=%d, size_n=%d not compatible"
                " with Marlin tile constraints (128, 64). Using Python dequant.",
                input_size, output_size,
            )
            # Keep weight_packed as-is for Python dequant
            self._use_python_dequant = True
            return
        
        self._use_python_dequant = False
        # Rename weight_packed to weight that marlin expects
        layer.weight = Parameter(layer.weight_packed.data, requires_grad=False)
        del layer.weight_packed
        # ct stores the inverse of what is expected by the marlin kernel
        layer.weight_global_scale = Parameter(
            1.0 / layer.weight_global_scale.max().to(torch.float32), requires_grad=False
        )

        prepare_fp4_layer_for_marlin(layer)'''

# Also add logger import
old_import = 'import torch\nfrom torch.nn.parameter import Parameter'
new_import = 'import torch\nfrom torch.nn.parameter import Parameter\n\nfrom vllm.logger import init_logger\nlogger = init_logger(__name__)'

if old_import in content:
    content = content.replace(old_import, new_import)

if old in content:
    content = content.replace(old, new)
    print("Applied Marlin shape guard")
else:
    print("Pattern not found in file")

with open(path, "w") as f:
    f.write(content)
print("Done.")
