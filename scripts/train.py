"""QLoRA fine-tuning for ZAYA1-8B tool-calling.

Trains attention projection LoRA adapters on structured tool-call
trajectories so ZAYA1-8B can drive the Godspeed agent loop.

Usage:
    python scripts/train.py --data data/train.jsonl --config configs/lora_tool_call.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune ZAYA1-8B for tool calling")
    parser.add_argument("--config", default="configs/lora_tool_call.yaml")
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--output", default="checkpoints/zaya1-tool-call")
    parser.add_argument("--dry-run", action="store_true", help="One batch, no save")
    parser.add_argument("--resume", help="Resume from checkpoint")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset(path: str) -> list[dict]:
    data: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info("Loaded %d training examples", len(data))
    return data


def format_chatml(example: dict, tokenizer) -> str:
    messages = example["messages"]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


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

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset

    model_cfg = cfg["model"]
    quant_cfg = cfg["quantization"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    logger.info("Loading tokenizer: %s", model_cfg["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    logger.info("Loading model: %s", model_cfg["name_or_path"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=getattr(torch, model_cfg["torch_dtype"]),
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    logger.info("Attaching LoRA adapters")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    logger.info("Loading dataset: %s", args.data)
    raw_data = load_dataset(args.data)
    if not raw_data:
        logger.error("No training data found")
        sys.exit(1)

    def tokenize_fn(examples):
        texts = [format_chatml(ex, tokenizer) for ex in examples["messages"]]
        return tokenizer(
            texts,
            truncation=True,
            max_length=train_cfg["max_seq_length"],
            padding=False,
        )

    dataset = Dataset.from_list([{"messages": ex} for ex in raw_data])
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        num_train_epochs=train_cfg["num_train_epochs"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        bf16=train_cfg["bf16"],
        fp16=train_cfg["fp16"],
        optim=train_cfg["optim"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        max_steps=1 if args.dry_run else -1,
        report_to=[],
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    if args.dry_run:
        logger.info("=== DRY RUN (1 step) ===")

    trainer.train(resume_from_checkpoint=args.resume)

    if not args.dry_run:
        logger.info("Saving adapter to %s", args.output)
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        logger.info("Training complete")


if __name__ == "__main__":
    main()
