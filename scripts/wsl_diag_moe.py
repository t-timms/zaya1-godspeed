import sys, os
sys.path.insert(0, "/home/ttimm/vllm-src")

def main():
    from vllm import LLM
    import torch
    model_path = "/mnt/c/Users/ttimm/Documents/Project Portfolio/zaya1-godspeed/zaya1-8b-nvfp4-ct-gs16"
    llm = LLM(
        model=model_path, dtype="bfloat16", trust_remote_code=True,
        max_model_len=256, gpu_memory_utilization=0.85, enforce_eager=True,
    )
    # Check first MoE layer
    for name, module in llm.llm_engine.model_executor.driver_worker.model_runner.model.named_modules():
        if "fused_moe" in name or "experts" in name:
            print(f"Module: {name}")
            if hasattr(module, 'quant_method'):
                qm = module.quant_method
                print(f"  quant_method type: {type(qm).__name__}")
                if hasattr(qm, '_w13_scale'):
                    print(f"  _w13_scale shape: {qm._w13_scale.shape}")
                    print(f"  _w13_scale min: {qm._w13_scale.float().min().item():.6f}")
                    print(f"  _w13_scale max: {qm._w13_scale.float().max().item():.6f}")
                if hasattr(qm, '_w2_scale'):
                    print(f"  _w2_scale shape: {qm._w2_scale.shape}")
                    print(f"  _w2_scale min: {qm._w2_scale.float().min().item():.6f}")
                if hasattr(module, 'w13_weight'):
                    w = module.w13_weight
                    print(f"  w13_weight dtype: {w.dtype} shape: {w.shape}")
                    print(f"  w13_weight min/max: {w.float().min().item():.2f}/{w.float().max().item():.2f}")
            break

if __name__ == "__main__":
    main()
