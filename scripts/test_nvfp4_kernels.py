"""Verify NVFP4 SM120 kernels are compiled and accessible."""

import sys

import torch

sys.path.insert(0, "/home/ttimm/vllm-src")

from vllm import _C_stable_libtorch

print("=== NVFP4 SM120 Kernel Test ===")
print()

# Try to directly access the ops via torch.ops._C
# PyTorch lazily loads op definitions, so accessing an unknown attr
# triggers the load
print("Testing torch.ops._C ops:")

test_ops = [
    "cutlass_scaled_mm_supports_fp4",
    "cutlass_scaled_fp4_mm",
    "cutlass_fp4_group_mm",
    "cutlass_scaled_mm",
]

for op_name in test_ops:
    try:
        # Direct attribute access (triggers lazy loading)
        op_fn = getattr(torch.ops._C, op_name)
        print(f"  torch.ops._C.{op_name}: EXISTS (type={type(op_fn).__name__})")
        if op_name == "cutlass_scaled_mm_supports_fp4":
            result = op_fn(120)
            print(f"    cutlass_scaled_mm_supports_fp4(120) = {result}")
    except RuntimeError as e:
        if "not defined" in str(e):
            print(f"  torch.ops._C.{op_name}: NOT DEFINED (may not be compiled)")
        else:
            print(f"  torch.ops._C.{op_name}: ERROR - {e}")
    except AttributeError:
        print(f"  torch.ops._C.{op_name}: NOT FOUND")
    except Exception as e:
        print(f"  torch.ops._C.{op_name}: UNEXPECTED ERROR - {type(e).__name__}: {e}")

print()

# Also check direct module attributes
print("Checking _C_stable_libtorch module for fp4 functions:")
for attr_name in sorted(dir(_C_stable_libtorch)):
    if "fp4" in attr_name.lower() or "nvfp4" in attr_name.lower():
        print(f"  _C_stable_libtorch.{attr_name}")

print()
print("DONE")
