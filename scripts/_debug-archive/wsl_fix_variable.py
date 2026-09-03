path = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
c = open(path).read()
c = c.replace("is_group_size_16", "is_group_size_valid")
open(path, "w").write(c)
print("FIXED")
