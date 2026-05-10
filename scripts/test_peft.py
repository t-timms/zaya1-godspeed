"""PEFT LoRA compatibility gate for ZAYA1-8B.

Verifies that the Zyphra transformers fork can load ZAYA1-8B
and that PEFT LoRA adapters attach correctly with gradient flow.

Usage:
    python scripts/test_peft.py [--model-id Zyphra/ZAYA1-8B]

Exit: 0 = gate passed, 1 = gate failed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def check_imports() -> bool:
    """Verify all required packages are importable."""
    ok = True
    for pkg in ("torch", "transformers", "peft", "bitsandbytes"):
        try:
            __import__(pkg)
            logger.info("  %s: OK", pkg)
        except ImportError:
            logger.error("  %s: MISSING — install with pip install %s", pkg, pkg)
            ok = False
    return ok


def check_cuda() -> bool:
    """Verify CUDA is available and report device."""
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return False
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info("GPU: %s (%.1f GB VRAM)", name, vram)
    return True


def check_architecture(model_id: str) -> bool:
    """Verify the model config is readable and reports correct architecture."""
    from transformers import AutoConfig

    logger.info("Loading config from %s ...", model_id)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    logger.info("  model_type: %s", config.model_type)
    logger.info("  architectures: %s", config.architectures)
    logger.info("  hidden_size: %s", config.hidden_size)
    logger.info("  num_hidden_layers: %s", config.num_hidden_layers)

    if config.model_type != "zaya":
        logger.error("Expected model_type='zaya', got '%s'", config.model_type)
        return False
    return True


def check_peft_attach(model_id: str, load_in_4bit: bool) -> bool:
    """Load model, attach LoRA, verify gradient flow."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    kwargs: dict = dict(device_map="auto", trust_remote_code=True)
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    logger.info("Loading model (%s) ...", "NF4" if load_in_4bit else "bf16")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    vram_base = torch.cuda.max_memory_allocated() / 1e9
    logger.info("VRAM base: %.2f GB", vram_base)

    # Discover attention projection modules for LoRA targeting
    target_candidates: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            for pattern in ("o_proj", "linear_q", "linear_k", "val_proj1", "val_proj2"):
                if name.endswith(pattern) and pattern not in target_candidates:
                    target_candidates.append(pattern)

    logger.info("LoRA target modules: %s", target_candidates)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_candidates,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    logger.info("Attaching LoRA ...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    vram_lora = torch.cuda.max_memory_allocated() / 1e9
    logger.info("VRAM with LoRA: %.2f GB (+%.2f GB)", vram_lora, vram_lora - vram_base)

    # Forward pass to verify gradient flow
    logger.info("Running forward pass + backward ...")
    msg = [{"role": "user", "content": "What is 2 + 2?"}]
    inp = tokenizer.apply_chat_template(
        msg,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    out = model(input_ids=inp, labels=inp)
    out.loss.backward()

    grad_count = sum(1 for _n, p in model.named_parameters() if p.grad is not None)
    logger.info("Params with gradients: %d", grad_count)

    if grad_count == 0:
        logger.error("No gradients — LoRA not hooked properly")
        return False

    # Token generation smoke test
    logger.info("Running generation smoke test ...")
    model.eval()
    with torch.no_grad():
        gen = model.generate(
            inp,
            max_new_tokens=32,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(gen[0][inp.shape[-1] :], skip_special_tokens=True)
    logger.info("Generated: %s", repr(text[:200]))

    logger.info("PEFT GATE: PASSED")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="PEFT compatibility gate for ZAYA1-8B")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("ZAYA_MODEL_ID", "Zyphra/ZAYA1-8B"),
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load in bf16 instead of NF4 (needs ~17 GB VRAM)",
    )
    args = parser.parse_args()

    logger.info("=== ZAYA1-8B PEFT Compatibility Gate ===")
    logger.info("Model: %s", args.model_id)

    checks = [
        ("Imports", check_imports),
        ("CUDA", check_cuda),
        ("Architecture", lambda: check_architecture(args.model_id)),
        ("PEFT attach", lambda: check_peft_attach(args.model_id, not args.no_4bit)),
    ]

    failed = False
    for name, fn in checks:
        logger.info("--- %s ---", name)
        try:
            if not fn():
                logger.error("%s: FAILED", name)
                failed = True
            else:
                logger.info("%s: OK", name)
        except Exception as exc:
            logger.exception("%s: ERROR — %s", name, exc)
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
