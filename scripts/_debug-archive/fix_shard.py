p = "/home/ttimm/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/zaya.py"
c = open(p).read()
c = c.replace("chkpt_weight_name, None, expert_id", 'chkpt_weight_name, "w1", expert_id')
c = c.replace('shard_id = None if "_packed" in param_name else "w2"', 'shard_id = "w2"')
open(p, "w").write(c)
print("FIXED shard_id values")
