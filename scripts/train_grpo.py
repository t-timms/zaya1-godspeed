"""GRPO policy improvement for ZAYA1-8B tool-calling (Phase 5 Stage 2).

After SFT on verified teacher trajectories, GRPO refines tool-calling
behavior via verifiable rewards: mechanical verify (primary signal),
format compliance, and schema validity.

Uses TRL GRPOTrainer with vLLM colocate mode for fast generation
on a single RTX 5070 Ti (16 GB).

Usage:
    python scripts/train_grpo.py \\
        --data data/train_zaya.jsonl \\
        --adapter checkpoints/zaya1-tool-call \\
        --output checkpoints/zaya1-tool-call-grpo
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ZAYA_TOOL_CALL_RE = re.compile(r"<zyphra_tool_call>(.*?)</zyphra_tool_call>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO fine-tune ZAYA1-8B for tool calling")
    parser.add_argument("--config", default="configs/lora_tool_call.yaml")
    parser.add_argument("--data", default="data/train_zaya.jsonl")
    parser.add_argument("--adapter", help="SFT adapter path to resume from")
    parser.add_argument("--output", default="checkpoints/zaya1-tool-call-grpo")
    parser.add_argument("--dry-run", action="store_true", help="One step, no save")
    parser.add_argument("--resume", help="Resume from GRPO checkpoint")
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


def format_reward(completions, **kwargs):
    """Reward for correct ZAYA XML tool-call format and think tags."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        score = 0.0
        if "<zyphra_tool_call>" in content and "</zyphra_tool_call>" in content:
            score += 0.3
            match = _ZAYA_TOOL_CALL_RE.search(content)
            if match:
                try:
                    tc = json.loads(match.group(1))
                    if "name" in tc and "arguments" in tc:
                        score += 0.2
                except json.JSONDecodeError:
                    pass
        if "<think>" in content and "</think>" in content:
            score += 0.2
        if "<zyphra_tool_response>" in content and "</zyphra_tool_response>" in content:
            score += 0.3
        rewards.append(score)
    return rewards


def schema_reward(completions, **kwargs):
    """Reward for valid JSON arguments in tool calls."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        score = 0.0
        for match in _ZAYA_TOOL_CALL_RE.finditer(content):
            try:
                tc = json.loads(match.group(1))
                if isinstance(tc.get("arguments"), dict):
                    score += 0.5
            except json.JSONDecodeError:
                pass
        rewards.append(min(score, 1.0))
    return rewards


def length_penalty(completions, **kwargs):
    """Small penalty for excessively long completions."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        if len(content) > 4096:
            rewards.append(-0.1)
        else:
            rewards.append(0.0)
    return rewards


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    check_vram()

    from patches.apply_zaya_patches import apply_all_patches

    apply_all_patches()

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    model_cfg = cfg["model"]
    quant_cfg = cfg["quantization"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg.get("training", {})

    raw_data = load_jsonl(args.data)
    if not raw_data:
        logger.error("No training data found")
        sys.exit(1)

    prompts = []
    for ex in raw_data:
        messages = ex["messages"]
        system_msgs = [m for m in messages if m["role"] == "system"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        if len(assistant_msgs) > 1:
            prompt_msgs = system_msgs + user_msgs + assistant_msgs[:-1]
        else:
            prompt_msgs = system_msgs + user_msgs
        prompts.append({"prompt": prompt_msgs})

    dataset = Dataset.from_list(prompts)

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
        lora_dropout=0.0,
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        use_rslora=lora_cfg.get("use_rslora", False),
        ensure_weight_tying=True,
    )

    training_args = GRPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5.0e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=1,
        max_prompt_length=2048,
        max_completion_length=2048,
        num_generations=4,
        beta=0.0,
        loss_type="dapo",
        scale_rewards="batch",
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        use_liger_kernel=train_cfg.get("use_liger_kernel", False),
        chat_template_kwargs={"enable_thinking": True},
        max_steps=1 if args.dry_run else -1,
        report_to=[],
        use_vllm=torch.cuda.is_available(),
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.30,
        model_init_kwargs={
            "trust_remote_code": True,
            "quantization_config": bnb_config,
            "device_map": "auto",
            "torch_dtype": getattr(torch, model_cfg["torch_dtype"]),
        },
    )

    model_id = args.adapter if args.adapter else model_cfg["name_or_path"]

    trainer = GRPOTrainer(
        model=model_id,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[format_reward, schema_reward, length_penalty],
        reward_weights=[0.5, 0.4, 0.1],
        peft_config=peft_config,
    )

    if args.dry_run:
        logger.info("=== GRPO DRY RUN (1 step) ===")

    trainer.train(resume_from_checkpoint=args.resume)

    if not args.dry_run:
        logger.info("Saving GRPO adapter to %s", args.output)
        trainer.save_model(args.output)
        logger.info("GRPO training complete")


if __name__ == "__main__":
    main()
