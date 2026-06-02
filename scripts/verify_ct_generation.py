"""Full generation quality test: decompress CT NVFP4 model, run inference, verify output."""

from __future__ import annotations

import logging
import sys
import time

import safetensors.torch as st
import torch
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.quantization import preset_name_to_scheme

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("gencheck")

CT_PATH = "zaya1-8b-nvfp4-ct/model.safetensors"


def main() -> int:
    scheme = preset_name_to_scheme("NVFP4A16", targets=["Linear"])

    log.info("Loading original BF16 model (tokenizer + config)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    orig = AutoModelForCausalLM.from_pretrained(
        "Zyphra/ZAYA1-8B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained("Zyphra/ZAYA1-8B", trust_remote_code=True)

    log.info("Loading CT NVFP4 state dict...")
    ct_state = st.load_file(CT_PATH, device="cpu")

    # Decompress all packed weights and patch into original model
    log.info("Decompressing all packed layers...")
    t0 = time.time()
    patched = 0
    for name, module in orig.named_modules():
        base = f"{name}.weight_packed"
        if base not in ct_state:
            continue
        d = {}
        for k in ("weight_packed", "weight_scale", "weight_global_scale"):
            key = f"{name}.{k}"
            if key in ct_state:
                d[k] = ct_state[key].to("cuda:0")
        if not d:
            continue
        decomp = NVFP4PackedCompressor.decompress(d, scheme)
        module.weight.data = decomp["weight"].to(torch.bfloat16).cpu()
        patched += 1

    log.info("Decompressed %d layers in %.0fs", patched, time.time() - t0)

    # Move model to GPU for generation
    log.info("Moving model to GPU...")
    orig = orig.to("cuda:0")
    vram = torch.cuda.max_memory_allocated() / 1e9
    log.info("VRAM: %.1f GB", vram)

    # Generate test prompts
    prompts = [
        "Write a Python function is_prime(n: int) -> bool that returns True if n is prime.",
        "Explain what a binary search tree is in one sentence.",
        "If x^2 + 5x + 6 = 0, solve for x step by step.",
    ]

    log.info("=" * 60)
    for i, p in enumerate(prompts):
        inp = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=True,
        ).to("cuda:0")

        with torch.no_grad():
            out = orig.generate(
                inp, max_new_tokens=150, do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
            )
        text = tok.decode(out[0][inp.shape[-1] :], skip_special_tokens=False)

        has_think = "</think>" in text
        has_im_end = "<|im_end|>" in text
        coherent = len(text.strip()) > 40
        ok = coherent and has_think and has_im_end

        log.info(
            "[%d/%d] %s | think=%s im_end=%s len=%d",
            i + 1,
            len(prompts),
            "PASS" if ok else "FAIL",
            has_think,
            has_im_end,
            len(text.strip()),
        )
        log.info("  %s", text[:300].replace("\n", " "))

    del orig
    torch.cuda.empty_cache()
    log.info("=" * 60)
    log.info("NVFP4 model generation test: DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
