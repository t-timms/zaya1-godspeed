"""Check QuantizationStrategy values."""
from compressed_tensors.quantization import QuantizationStrategy

print("QuantizationStrategy values:")
for s in QuantizationStrategy:
    print(f"  {s.name} = {s.value}")
