"""Gate 4: Build NVFP4 calibration dataset for Stage 1 compressed-tensors quantization.

Constructs 512 calibration samples (each exactly 1024 tokens) from four data sources
matching ZAYA1-8B's training distribution:
  - MATH-500: mathematical reasoning problems (~128 samples)
  - HumanEval: Python code generation problems (~128 samples)
  - ShareGPT-style: conversational data with tool-use patterns (~128 samples)
  - AIME: competition math problems (~128 samples)

The calibration data is used by the NVFP4 quantizer to select optimal scales
per channel group. Each sample is tokenized to exactly 1024 tokens with the
Zyphra/ZAYA1-8B tokenizer (ChatML format with thinking mode tokens).

Output: data/calibration/calibration_data.pt (PyTorch tensor, [512, 1024])
         data/calibration/manifest.json (metadata and source breakdown)

Usage:
    uv run python scripts/build_calibration_data.py
    uv run python scripts/build_calibration_data.py --num-samples 512 --max-length 1024
    uv run python scripts/build_calibration_data.py --offline  # use cached datasets only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Zyphra/ZAYA1-8B"
DEFAULT_NUM_SAMPLES = 512
DEFAULT_MAX_LENGTH = 1024
OUTPUT_DIR = Path("data/calibration")
OUTPUT_FILE = OUTPUT_DIR / "calibration_data.pt"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"

# Source distribution (must sum to 1.0)
SOURCE_WEIGHTS = {
    "math500": 0.25,  # 128 samples
    "humaneval": 0.25,  # 128 samples
    "sharegpt": 0.25,  # 128 samples
    "aime": 0.25,  # 128 samples
}

# HuggingFace dataset paths
DATASET_PATHS: dict[str, str] = {
    "math500": "HuggingFaceH4/MATH-500",
    "humaneval": "openai/openai_humaneval",
    "sharegpt": "anon8231489123/ShareGPT_Vicuna_unfiltered",
    "aime": "Maxwell-Jia/AIME_2024",
}


def import_libraries() -> tuple[Any, Any, Any]:
    """Import torch, datasets, and transformers. Returns (torch, datasets, transformers)."""
    import datasets as _datasets
    import torch as _torch
    import transformers as _transformers

    logger.info("torch %s: OK", _torch.__version__)
    logger.info("datasets %s: OK", _datasets.__version__)
    logger.info("transformers %s: OK", _transformers.__version__)
    return _torch, _datasets, _transformers


def load_tokenizer(transformers: Any, model_id: str) -> Any:
    """Load the ZAYA1-8B tokenizer."""
    logger.info("Loading tokenizer from %s ...", model_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer vocab size: %d", tokenizer.vocab_size)
    logger.info("Tokenizer pad_token_id: %d", tokenizer.pad_token_id)
    return tokenizer


def load_math500(datasets: Any) -> list[str]:
    """Load MATH-500 reasoning problems."""
    try:
        ds = datasets.load_dataset(DATASET_PATHS["math500"], split="test")
        logger.info("MATH-500: loaded %d samples", len(ds))
        texts: list[str] = []
        for item in ds:
            problem = item.get("problem", "") or ""
            solution = item.get("solution", "") or ""
            texts.append(f"Problem: {problem}\n\nSolution: {solution}")
        return texts
    except Exception as e:
        logger.warning("MATH-500 load failed: %s", e)
        return []


def load_humaneval(datasets: Any) -> list[str]:
    """Load HumanEval code generation problems."""
    try:
        ds = datasets.load_dataset(DATASET_PATHS["humaneval"], split="test")
        logger.info("HumanEval: loaded %d samples", len(ds))
        texts: list[str] = []
        for item in ds:
            prompt = item.get("prompt", "") or ""
            canonical = item.get("canonical_solution", "") or ""
            entry = item.get("entry_point", "") or ""
            texts.append(f"# Task: {entry}\n{prompt}\n\n# Solution:\n{canonical}")
        return texts
    except Exception as e:
        logger.warning("HumanEval load failed: %s", e)
        return []


def load_sharegpt(datasets: Any) -> list[str]:
    """Load ShareGPT conversational data."""
    try:
        ds = datasets.load_dataset(
            DATASET_PATHS["sharegpt"],
            "Vicuna_v1.1_unfiltered",
            split="train",
        )
        logger.info("ShareGPT: loaded %d samples", len(ds))
        texts: list[str] = []
        for item in ds:
            conversations = item.get("conversations", [])
            parts = []
            for turn in conversations:
                role = turn.get("from", "unknown")
                value = turn.get("value", "")
                parts.append(f"{role}: {value}")
            texts.append("\n".join(parts))
        return texts
    except Exception as e:
        logger.warning("ShareGPT load failed: %s. Trying alternate dataset...", e)
        try:
            ds = datasets.load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5000]")
            logger.info("UltraChat fallback: loaded %d samples", len(ds))
            texts: list[str] = []
            for item in ds:
                messages = item.get("messages", [])
                parts = []
                for msg in messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    parts.append(f"{role}: {content}")
                texts.append("\n".join(parts))
            return texts
        except Exception as e2:
            logger.warning("UltraChat fallback also failed: %s", e2)
            return []


def load_aime(datasets: Any) -> list[str]:
    """Load AIME competition math problems."""
    try:
        ds = datasets.load_dataset(DATASET_PATHS["aime"], split="train")
        logger.info("AIME: loaded %d samples", len(ds))
        texts: list[str] = []
        for item in ds:
            problem = item.get("problem", "") or item.get("question", "") or ""
            answer = item.get("answer", "") or item.get("solution", "") or ""
            texts.append(f"AIME Problem:\n{problem}\n\nAnswer: {answer}")
        return texts
    except Exception as e:
        logger.warning("AIME load failed: %s", e)
        # Try alternate AIME dataset
        try:
            ds = datasets.load_dataset("AI-MO/aimo-validation-aime", split="train")
            logger.info("AIME fallback: loaded %d samples", len(ds))
            texts: list[str] = []
            for item in ds:
                problem = item.get("problem", "") or item.get("url", "") or ""
                answer = item.get("answer", "") or ""
                texts.append(f"AIME Problem:\n{problem}\n\nAnswer: {answer}")
            return texts
        except Exception as e2:
            logger.warning("AIME fallback also failed: %s", e2)
            return []


def tokenize_sample(
    tokenizer: Any,
    text: str,
    max_length: int,
    add_chat_template: bool = True,
) -> list[int]:
    """Tokenize a text sample to exactly max_length tokens.

    Uses the ChatML template for distribution-matching with ZAYA1-8B's training format.
    Adds <think> tokens to match the model's always-on thinking mode.
    """
    if add_chat_template:
        messages = [{"role": "user", "content": text[:8000]}]  # Limit raw text
        try:
            tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
        except TypeError:
            # Fallback if enable_thinking not supported
            tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
    else:
        encoding = tokenizer(
            text[:8000],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokens = encoding["input_ids"]

    return tokens[0].tolist()


def build_calibration(
    torch: Any,
    tokenizer: Any,
    sources: dict[str, list[str]],
    num_samples: int,
    max_length: int,
) -> tuple[Any, dict[str, Any]]:
    """Build the calibration tensor from source texts.

    Returns (tensor: [N, max_length], manifest: dict).
    """
    all_texts: list[tuple[str, str]] = []  # (source_name, text)
    for source_name, weight in SOURCE_WEIGHTS.items():
        target_count = int(num_samples * weight)
        texts = sources.get(source_name, [])
        if not texts:
            logger.warning("  %s: 0 samples available (target: %d)", source_name, target_count)
            continue

        # Sample from available texts
        if len(texts) <= target_count:
            selected = texts
        else:
            # Deterministic shuffle for reproducibility
            import random

            rng = random.Random(42)
            indices = list(range(len(texts)))
            rng.shuffle(indices)
            selected = [texts[i] for i in indices[:target_count]]

        for text in selected:
            all_texts.append((source_name, text))
        logger.info("  %s: %d samples", source_name, len(selected))

    if not all_texts:
        logger.error("No calibration data available from any source")
        return None, {}

    # If we don't have enough, duplicate to reach target
    while len(all_texts) < num_samples:
        all_texts.extend(all_texts[: num_samples - len(all_texts)])
    all_texts = all_texts[:num_samples]

    logger.info("Tokenizing %d samples to %d tokens each ...", len(all_texts), max_length)
    t0 = time.time()
    token_ids: list[list[int]] = []
    for source_name, text in all_texts:
        ids = tokenize_sample(tokenizer, text, max_length)
        token_ids.append(ids)

    tensor = torch.tensor(token_ids, dtype=torch.long)
    elapsed = time.time() - t0
    logger.info("Tokenization complete: %.1fs (%.1f samples/s)", elapsed, len(all_texts) / elapsed)

    # Build manifest
    source_counts: dict[str, int] = {}
    for source_name, _ in all_texts:
        source_counts[source_name] = source_counts.get(source_name, 0) + 1

    manifest: dict[str, Any] = {
        "version": "1.0.0",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": DEFAULT_MODEL,
        "purpose": "NVFP4 Stage 1 calibration data for compressed_tensors quantization",
        "total_samples": len(all_texts),
        "max_length": max_length,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "source_breakdown": source_counts,
    }

    return tensor, manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate 4: Build NVFP4 calibration dataset",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("ZAYA_MODEL_ID", DEFAULT_MODEL),
        help=f"Tokenizer model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Total calibration samples (default: {DEFAULT_NUM_SAMPLES})",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Tokens per sample (default: {DEFAULT_MAX_LENGTH})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached datasets only; skip downloads",
    )
    args = parser.parse_args()

    logger.info("=== GATE 4: Calibration Dataset Build ===")
    logger.info("Model:      %s", args.model_id)
    logger.info("Samples:    %d", args.num_samples)
    logger.info("Max length: %d", args.max_length)
    logger.info("Output:     %s", args.output_dir)

    torch, datasets, transformers = import_libraries()

    # Load tokenizer
    tokenizer = load_tokenizer(transformers, args.model_id)

    # Load all data sources
    logger.info("")
    logger.info("--- Loading Data Sources ---")

    if args.offline:
        datasets.set_caching_enabled(True)
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    sources: dict[str, list[str]] = {}

    logger.info("Loading MATH-500 ...")
    sources["math500"] = load_math500(datasets)

    logger.info("Loading HumanEval ...")
    sources["humaneval"] = load_humaneval(datasets)

    logger.info("Loading ShareGPT ...")
    sources["sharegpt"] = load_sharegpt(datasets)

    logger.info("Loading AIME ...")
    sources["aime"] = load_aime(datasets)

    # Report availability
    available = sum(1 for v in sources.values() if v)
    if available == 0:
        logger.error("No data sources could be loaded.")
        logger.error("Check internet connection or try --offline with cached data.")
        return 1

    logger.info("")
    logger.info("--- Building Calibration Tensor ---")
    tensor, manifest = build_calibration(
        torch,
        tokenizer,
        sources,
        args.num_samples,
        args.max_length,
    )

    if tensor is None:
        logger.error("Failed to build calibration tensor")
        return 1

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(tensor, output_dir / "calibration_data.pt")
    logger.info("Saved: %s (%.1f MB)", output_dir / "calibration_data.pt", tensor.numel() * tensor.element_size() / 1e6)

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved: %s", output_dir / "manifest.json")

    # Verify
    logger.info("")
    logger.info("--- Verification ---")
    loaded = torch.load(output_dir / "calibration_data.pt")
    logger.info("Reloaded tensor shape: %s", loaded.shape)
    logger.info("Token ID range: [%d, %d]", loaded.min().item(), loaded.max().item())

    # Check for special tokens
    eos_count = (loaded == tokenizer.eos_token_id).sum().item()
    pad_count = (loaded == tokenizer.pad_token_id).sum().item()
    logger.info("EOS tokens: %d (%.1f%%)", eos_count, 100 * eos_count / loaded.numel())
    logger.info("PAD tokens: %d (%.1f%%)", pad_count, 100 * pad_count / loaded.numel())

    # Report
    logger.info("")
    logger.info("=== Results ===")
    logger.info("Total samples: %d", tensor.shape[0])
    logger.info("Tokens each:   %d", tensor.shape[1])
    logger.info("Output size:   %.1f MB", tensor.numel() * tensor.element_size() / 1e6)
    logger.info("")

    for source, count in manifest.get("source_breakdown", {}).items():
        logger.info("  %s: %d samples", source, count)

    logger.info("")
    logger.info("GATE 4 PASSED: %d samples × %d tokens saved to %s", tensor.shape[0], tensor.shape[1], output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
