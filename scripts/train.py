"""QLoRA fine-tuning for ZAYA1-8B tool-calling.

Trains attention projection LoRA adapters on structured tool-call
trajectories so ZAYA1-8B can drive the Godspeed agent loop.

Uses TRL SFTTrainer with SFTConfig for SOTA training: assistant_only_loss,
packing, native conversational dataset format, and automatic chat template
application.

Usage:
    python scripts/train.py --data data/train_zaya.jsonl --config configs/lora_tool_call.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune ZAYA1-8B for tool calling")
    parser.add_argument("--config", default="configs/lora_tool_call.yaml")
    parser.add_argument("--data", default="data/train_zaya.jsonl")
    parser.add_argument("--output", default="checkpoints/zaya1-tool-call")
    parser.add_argument("--dry-run", action="store_true", help="One batch, no save")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--eval-data", help="Optional eval dataset JSONL")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    data: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info("Loaded %d examples from %s", len(data), path)
    return data


def check_vram() -> None:
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        sys.exit(1)
    free, total = torch.cuda.mem_get_info()
    free_gb = free / 1e9
    total_gb = total / 1e9
    logger.info("VRAM: %.1f GB free / %.1f GB total", free_gb, total_gb)
    if free_gb < 12:
        logger.warning("Less than 12 GB free — may OOM")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    check_vram()

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model_cfg = cfg["model"]
    quant_cfg = cfg["quantization"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    logger.info("Loading training data: %s", args.data)
    raw_data = load_jsonl(args.data)
    if not raw_data:
        logger.error("No training data found")
        sys.exit(1)

    train_dataset = Dataset.from_list(raw_data)

    eval_dataset = None
    if args.eval_data:
        eval_raw = load_jsonl(args.eval_data)
        if eval_raw:
            eval_dataset = Dataset.from_list(eval_raw)

    from peft import LoraConfig, TaskType
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        use_rslora=lora_cfg.get("use_rslora", False),
        init_lora_weights=lora_cfg.get("init_lora_weights", "default"),
        ensure_weight_tying=True,
    )

    training_args = SFTConfig(
        output_dir=args.output,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        num_train_epochs=train_cfg["num_train_epochs"],
        max_seq_length=train_cfg["max_seq_length"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        bf16=train_cfg["bf16"],
        fp16=train_cfg["fp16"],
        optim=train_cfg["optim"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        loss_type=train_cfg.get("loss_type", "nll"),
        use_liger_kernel=train_cfg.get("use_liger_kernel", False),
        chat_template_kwargs={"enable_thinking": True},
        max_steps=1 if args.dry_run else -1,
        report_to=[],
        packing=train_cfg.get("packing", False),
        assistant_only_loss=True,
        model_init_kwargs={
            "trust_remote_code": True,
            "quantization_config": bnb_config,
            "device_map": "auto",
            "torch_dtype": getattr(torch, model_cfg["torch_dtype"]),
        },
    )

    trainer = SFTTrainer(
        model=model_cfg["name_or_path"],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    if args.dry_run:
        logger.info("=== DRY RUN (1 step) ===")

    trainer.train(resume_from_checkpoint=args.resume)

    if not args.dry_run:
        logger.info("Saving adapter to %s", args.output)
        trainer.save_model(args.output)
        logger.info("Training complete")


if __name__ == "__main__":
    main()
