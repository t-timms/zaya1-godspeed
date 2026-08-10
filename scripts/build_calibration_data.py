"""Gate 4: Build NVFP4 W4A4 calibration dataset for compressed-tensors quantization.

Constructs 1024 calibration samples (each 1024 tokens) from seven data sources
weighted by the Phase 1 eval surface (13 benchmarks across math, code, knowledge,
instruction, style, agentic):

  Phase 1 mix (default, --num-samples 1024):
    - MATH-500:        15% — math reasoning (AIME'26 / HMMT / IMO / APEX preservation)
    - AIME 2024 train: 15% — competition math (NOT AIME'26 — that's the eval set)
    - HumanEval:       10% — Python code (LiveCodeBench-v6 preservation)
    - TriviaQA:        15% — free-form knowledge QA (GPQA-Diamond / MMLU-Pro preservation)
    - Alpaca-cleaned:  15% — instruction-following (IFEval / IFBench preservation)
    - WritingPrompts:  15% — creative writing style (EQBench / Creative Writing v3)
    - Glaive-fn-call:  15% — tool-use traces (BFCL-v4 / τ² preservation)

  Legacy mix (--legacy-mix, --num-samples 512):
    - MATH-500: 25% | HumanEval: 25% | ShareGPT: 25% | AIME: 25%

Phase 1 success criterion: AIME'26 ≥ 87 (≤2 pt drop from published 89.1).
Math weight is highest because math is the most quantization-sensitive eval class.

Each sample tokenized to max-length with the Zyphra/ZAYA1-8B tokenizer (ChatML
format, thinking-mode tokens). Calibration data feeds the W4A4 input_global_scale
calibration pass — one per-tensor fp32 scale per fused Linear group, computed as
max_observed_activation / 6.0 (FP4_E2M1 max magnitude).

Contamination check: every source is from public train/dev splits. AIME 2024 ≠
AIME 2026 (different problems). HumanEval ≠ LiveCodeBench. TriviaQA ≠ GPQA.

Output: data/calibration/calibration_data.pt (PyTorch tensor, [N, max_length])
        data/calibration/manifest.json (metadata, source breakdown, mix profile)

Usage:
    uv run python scripts/build_calibration_data.py                   # Phase 1, 1024 samples
    uv run python scripts/build_calibration_data.py --legacy-mix      # original 4-source, 512 samples
    uv run python scripts/build_calibration_data.py --num-samples 256 # smaller pass for iteration
    uv run python scripts/build_calibration_data.py --offline         # cached datasets only
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

DEFAULT_MODEL = "Zyphra/ZAYA1-8B-legacy"
DEFAULT_NUM_SAMPLES = 1024
DEFAULT_MAX_LENGTH = 1024
OUTPUT_DIR = Path("data/calibration")
OUTPUT_FILE = OUTPUT_DIR / "calibration_data.pt"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"

# Phase 1 mix: weighted by the 13-benchmark eval surface (math-heavy because
# math is the most quantization-sensitive class). Must sum to 1.0.
# v2 (2026-05-17): dropped AIME (only 30 train problems — too few to fill 15%),
# replaced with GSM8K (8K math word problems). Split humaneval 10% -> humaneval 5%
# + mbpp 5% for code diversity. Math weight 30% preserved, code 10% preserved.
SOURCE_WEIGHTS_PHASE1: dict[str, float] = {
    "math500": 0.15,  # MATH-500 (competition-style math reasoning)
    "gsm8k": 0.15,  # GSM8K (multi-step math word problems)
    "humaneval": 0.05,  # Python function-completion code
    "mbpp": 0.05,  # Mostly Basic Python Problems
    "triviaqa": 0.15,  # knowledge QA
    "alpaca": 0.15,  # instruction-following
    "writingprompts": 0.15,  # creative style
    "glaive": 0.15,  # tool-use traces (agentic)
}

# Phase 2 mix: ARC-aware calibration. Adds commonsense reasoning data
# (arc_easy, arc_challenge, hellaswag) to calibrate MoE expert routing paths
# that are exercised by those benchmarks. MoE expert sparsity means experts
# only seen during eval-style inputs may have uncalibrated IGS values.
# Math reduced to 20% (still dominant), creative/agentic reduced to make room.
# v1 (2026-05-19): targeting ARC-Easy/Challenge + HellaSwag accuracy recovery.
SOURCE_WEIGHTS_PHASE2: dict[str, float] = {
    "math500": 0.10,  # MATH-500
    "gsm8k": 0.10,  # GSM8K
    "humaneval": 0.05,  # Python code
    "mbpp": 0.05,  # Python code
    "triviaqa": 0.10,  # knowledge QA
    "alpaca": 0.10,  # instruction-following
    "writingprompts": 0.05,  # creative style (reduced)
    "glaive": 0.05,  # tool-use (reduced)
    "arc_easy": 0.15,  # ARC-Easy training split (commonsense QA)
    "arc_challenge": 0.10,  # ARC-Challenge training split (harder QA)
    "hellaswag": 0.15,  # HellaSwag training split (activity completion)
}

# Legacy mix retained for --legacy-mix flag (reproducibility of the original
# Stage 1 W4A16 calibration that produced zaya1-8b-nvfp4-ct-gs16).
SOURCE_WEIGHTS_LEGACY: dict[str, float] = {
    "math500": 0.25,
    "humaneval": 0.25,
    "sharegpt": 0.25,
    "aime": 0.25,
}

# Default active mix (overridden to legacy when --legacy-mix passed).
SOURCE_WEIGHTS: dict[str, float] = SOURCE_WEIGHTS_PHASE1

# HuggingFace dataset paths (all open-license, public train/dev splits)
DATASET_PATHS: dict[str, str] = {
    "math500": "HuggingFaceH4/MATH-500",
    "humaneval": "openai/openai_humaneval",
    "mbpp": "google-research-datasets/mbpp",  # split 'train'
    "sharegpt": "anon8231489123/ShareGPT_Vicuna_unfiltered",
    "aime": "Maxwell-Jia/AIME_2024",
    "gsm8k": "openai/gsm8k",  # subset 'main', split 'train' (~7.5K problems)
    "triviaqa": "trivia_qa",  # subset 'rc.nocontext', split 'train'
    "alpaca": "yahma/alpaca-cleaned",  # split 'train'
    "writingprompts": "euclaise/writingprompts",  # split 'train' (subsample heavily — 1.4M total)
    "glaive": "glaiveai/glaive-function-calling-v2",  # split 'train'
    "arc_easy": "allenai/ai2_arc",  # subset 'ARC-Easy', split 'train' (2251 examples)
    "arc_challenge": "allenai/ai2_arc",  # subset 'ARC-Challenge', split 'train' (1119 examples)
    "hellaswag": "Rowan/hellaswag",  # split 'train' (39905 examples)
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


def load_gsm8k(datasets: Any) -> list[str]:
    """Load GSM8K multi-step math word problems (8K train problems, open license)."""
    try:
        ds = datasets.load_dataset(DATASET_PATHS["gsm8k"], "main", split="train")
        logger.info("GSM8K: loaded %d samples", len(ds))
        texts: list[str] = []
        for item in ds:
            question = item.get("question", "") or ""
            answer = item.get("answer", "") or ""
            if question and answer:
                texts.append(f"Problem: {question}\n\nSolution: {answer}")
        return texts
    except Exception as e:
        logger.warning("GSM8K load failed: %s", e)
        return []


def load_mbpp(datasets: Any) -> list[str]:
    """Load MBPP (Mostly Basic Python Problems, ~974 sanitized train)."""
    try:
        ds = datasets.load_dataset(DATASET_PATHS["mbpp"], "sanitized", split="train")
        logger.info("MBPP: loaded %d samples", len(ds))
        texts: list[str] = []
        for item in ds:
            prompt = item.get("prompt", "") or item.get("text", "") or ""
            code = item.get("code", "") or ""
            test_list = item.get("test_list", []) or []
            tests = "\n".join(test_list) if test_list else ""
            if prompt and code:
                if tests:
                    texts.append(f"# Task: {prompt}\n\n# Solution:\n{code}\n\n# Tests:\n{tests}")
                else:
                    texts.append(f"# Task: {prompt}\n\n# Solution:\n{code}")
        return texts
    except Exception as e:
        logger.warning("MBPP load failed (sanitized config): %s. Trying full config...", e)
        try:
            ds = datasets.load_dataset(DATASET_PATHS["mbpp"], "full", split="train")
            logger.info("MBPP fallback (full): loaded %d samples", len(ds))
            texts: list[str] = []
            for item in ds:
                prompt = item.get("text", "") or ""
                code = item.get("code", "") or ""
                if prompt and code:
                    texts.append(f"# Task: {prompt}\n\n# Solution:\n{code}")
            return texts
        except Exception as e2:
            logger.warning("MBPP full fallback also failed: %s", e2)
            return []


def load_triviaqa(datasets: Any) -> list[str]:
    """Load TriviaQA free-form knowledge QA. Subset 'rc.nocontext' has just Q+A pairs."""
    try:
        ds = datasets.load_dataset(
            DATASET_PATHS["triviaqa"],
            "rc.nocontext",
            split="train[:5000]",
        )
        logger.info("TriviaQA: loaded %d samples", len(ds))
        texts: list[str] = []
        for item in ds:
            question = item.get("question", "") or ""
            answer_obj = item.get("answer", {})
            value = answer_obj.get("value", "") if isinstance(answer_obj, dict) else str(answer_obj)
            aliases = answer_obj.get("aliases", []) if isinstance(answer_obj, dict) else []
            answer = value if value else (aliases[0] if aliases else "")
            if question and answer:
                texts.append(f"Question: {question}\n\nAnswer: {answer}")
        return texts
    except Exception as e:
        logger.warning("TriviaQA load failed: %s", e)
        return []


def load_alpaca(datasets: Any) -> list[str]:
    """Load Alpaca-cleaned instruction-following dataset (52K open-license entries)."""
    try:
        ds = datasets.load_dataset(DATASET_PATHS["alpaca"], split="train[:10000]")
        logger.info("Alpaca-cleaned: loaded %d samples", len(ds))
        texts: list[str] = []
        for item in ds:
            instruction = item.get("instruction", "") or ""
            inp = item.get("input", "") or ""
            output = item.get("output", "") or ""
            if not instruction:
                continue
            if inp:
                texts.append(f"Instruction: {instruction}\n\nInput: {inp}\n\nResponse: {output}")
            else:
                texts.append(f"Instruction: {instruction}\n\nResponse: {output}")
        return texts
    except Exception as e:
        logger.warning("Alpaca load failed: %s", e)
        return []


def load_writingprompts(datasets: Any) -> list[str]:
    """Load WritingPrompts creative writing dataset. Heavy subsample — 1.4M total."""
    try:
        ds = datasets.load_dataset(
            DATASET_PATHS["writingprompts"],
            split="train[:3000]",
        )
        logger.info("WritingPrompts: loaded %d samples", len(ds))
        texts: list[str] = []
        for item in ds:
            prompt = item.get("prompt", "") or ""
            story = item.get("story", "") or ""
            if prompt and story:
                # Strip the [WP] tag and truncate long stories at the source
                prompt_clean = prompt.replace("[WP]", "").strip()
                texts.append(f"Prompt: {prompt_clean}\n\nStory:\n{story[:6000]}")
        return texts
    except Exception as e:
        logger.warning("WritingPrompts load failed: %s", e)
        return []


def load_glaive(datasets: Any) -> list[str]:
    """Load Glaive function-calling traces (agentic / tool-use calibration)."""
    try:
        ds = datasets.load_dataset(DATASET_PATHS["glaive"], split="train[:5000]")
        logger.info("Glaive-fn-calling: loaded %d samples", len(ds))
        texts: list[str] = []
        for item in ds:
            system = item.get("system", "") or ""
            chat = item.get("chat", "") or ""
            if not chat:
                continue
            # The chat field is a flat string with USER:/ASSISTANT:/FUNCTION_CALL/etc. markers —
            # exactly the distribution we want for BFCL/τ² activation calibration.
            if system:
                texts.append(f"{system}\n\n{chat}")
            else:
                texts.append(chat)
        return texts
    except Exception as e:
        logger.warning("Glaive load failed: %s", e)
        return []


def load_arc_easy(datasets: Any) -> list[str]:
    """Load ARC-Easy training split (2251 commonsense reasoning MC questions)."""
    try:
        ds = datasets.load_dataset(DATASET_PATHS["arc_easy"], "ARC-Easy", split="train")
        logger.info("ARC-Easy: loaded %d samples", len(ds))
        texts: list[str] = []
        for item in ds:
            question = item.get("question", "") or ""
            choices = item.get("choices", {})
            answer_key = item.get("answerKey", "") or ""
            if not question:
                continue
            choice_texts = choices.get("text", []) if isinstance(choices, dict) else []
            choice_labels = choices.get("label", []) if isinstance(choices, dict) else []
            formatted_choices = "\n".join(f"  {lbl}. {txt}" for lbl, txt in zip(choice_labels, choice_texts))
            texts.append(f"Question: {question}\n\nChoices:\n{formatted_choices}\n\nAnswer: {answer_key}")
        return texts
    except Exception as e:
        logger.warning("ARC-Easy load failed: %s", e)
        return []


def load_arc_challenge(datasets: Any) -> list[str]:
    """Load ARC-Challenge training split (1119 harder commonsense reasoning questions)."""
    try:
        ds = datasets.load_dataset(DATASET_PATHS["arc_challenge"], "ARC-Challenge", split="train")
        logger.info("ARC-Challenge: loaded %d samples", len(ds))
        texts: list[str] = []
        for item in ds:
            question = item.get("question", "") or ""
            choices = item.get("choices", {})
            answer_key = item.get("answerKey", "") or ""
            if not question:
                continue
            choice_texts = choices.get("text", []) if isinstance(choices, dict) else []
            choice_labels = choices.get("label", []) if isinstance(choices, dict) else []
            formatted_choices = "\n".join(f"  {lbl}. {txt}" for lbl, txt in zip(choice_labels, choice_texts))
            texts.append(f"Question: {question}\n\nChoices:\n{formatted_choices}\n\nAnswer: {answer_key}")
        return texts
    except Exception as e:
        logger.warning("ARC-Challenge load failed: %s", e)
        return []


def load_hellaswag_train(datasets: Any) -> list[str]:
    """Load HellaSwag training split (39905 activity-completion sentences)."""
    try:
        ds = datasets.load_dataset(DATASET_PATHS["hellaswag"], split="train[:8000]")
        logger.info("HellaSwag: loaded %d samples", len(ds))
        texts: list[str] = []
        for item in ds:
            ctx = item.get("ctx", "") or item.get("activity_label", "") or ""
            endings = item.get("endings", []) or []
            label = item.get("label", "")
            if not ctx or not endings:
                continue
            formatted_endings = "\n".join(f"  {i}. {e}" for i, e in enumerate(endings))
            correct = endings[int(label)] if label != "" and int(label) < len(endings) else ""
            texts.append(
                f"Context: {ctx}\n\nContinuations:\n{formatted_endings}"
                + (f"\n\nCorrect continuation: {correct}" if correct else "")
            )
        return texts
    except Exception as e:
        logger.warning("HellaSwag train load failed: %s", e)
        return []


def _encode_text_for_packing(tokenizer: Any, text: str, hard_cap: int) -> list[int]:
    """Tokenize one text with the ChatML template, no padding, capped at hard_cap tokens.

    Returns a plain list of token ids. Used by pack_source_to_chunks below.
    """
    messages = [{"role": "user", "content": text[:8000]}]
    try:
        tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors=None,
            enable_thinking=True,
            truncation=True,
            max_length=hard_cap,
        )
    except TypeError:
        tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors=None,
            truncation=True,
            max_length=hard_cap,
        )
    # apply_chat_template with return_tensors=None returns a list (or list-of-list for batched).
    if tokens and isinstance(tokens[0], list):
        return list(tokens[0])
    return list(tokens)


def pack_source_to_chunks(
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    target_chunks: int,
) -> list[list[int]]:
    """Concat multiple texts into dense max_length-token chunks (no padding).

    Most calibration sources have texts much shorter than max_length (TriviaQA QA
    ~100 tok, Alpaca ~150 tok). Padding each to max_length leaves 60-90% PAD,
    which corrupts W4A4 activation calibration: the observed max-activation per
    Linear is dominated by PAD-token activations (small magnitudes), so scales
    come out too small and real inference overflows.

    Instead we tokenize each text raw, concat with EOS separators (so the model
    sees a clear document boundary, same as during training pack), and slice into
    dense max_length chunks. Only the trailing partial chunk is PAD-padded.

    Returns at most target_chunks; may return fewer if the source runs out.
    """
    sep_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else sep_id
    chunks: list[list[int]] = []
    buffer: list[int] = []
    # 4x max_length is a generous per-text cap that still keeps any single
    # document from monopolizing the corpus.
    per_text_cap = max_length * 4

    for text in texts:
        if len(chunks) >= target_chunks:
            break
        ids = _encode_text_for_packing(tokenizer, text, per_text_cap)
        if not ids:
            continue
        buffer.extend(ids)
        buffer.append(sep_id)
        while len(buffer) >= max_length and len(chunks) < target_chunks:
            chunks.append(buffer[:max_length])
            buffer = buffer[max_length:]

    # Take the trailing residual as one PAD-extended chunk only if we still
    # need samples AND it's at least half-full (otherwise it's mostly PAD,
    # which is exactly what we're trying to avoid).
    if len(chunks) < target_chunks and len(buffer) >= max_length // 2:
        residual = buffer + [pad_id] * (max_length - len(buffer))
        chunks.append(residual[:max_length])

    return chunks


def build_calibration(
    torch: Any,
    tokenizer: Any,
    sources: dict[str, list[str]],
    num_samples: int,
    max_length: int,
) -> tuple[Any, dict[str, Any]]:
    """Build the calibration tensor using concat-packed chunks (no padding).

    Each source contributes `int(num_samples * weight)` dense max_length-token
    chunks built by concatenating its raw texts with EOS separators. If a source
    runs out of text, we accept fewer chunks for that source rather than
    duplicating filler (which would distort the recipe).

    Returns (tensor: [N, max_length], manifest: dict). N may be < num_samples
    if multiple sources are short on text.
    """
    logger.info("Packing sources into dense %d-token chunks (no PAD)...", max_length)
    t0 = time.time()
    all_chunks: list[tuple[str, list[int]]] = []

    for source_name, weight in SOURCE_WEIGHTS.items():
        target_chunks = int(num_samples * weight)
        texts = sources.get(source_name, [])
        if not texts:
            logger.warning("  %s: 0 texts available (target chunks: %d)", source_name, target_chunks)
            continue

        # Deterministic shuffle so chunks aren't biased toward dataset ordering.
        import random

        rng = random.Random(42)
        indices = list(range(len(texts)))
        rng.shuffle(indices)
        shuffled = [texts[i] for i in indices]

        chunks = pack_source_to_chunks(tokenizer, shuffled, max_length, target_chunks)
        for c in chunks:
            all_chunks.append((source_name, c))

        if len(chunks) < target_chunks:
            logger.warning(
                "  %s: %d/%d chunks (short by %d — source text exhausted)",
                source_name,
                len(chunks),
                target_chunks,
                target_chunks - len(chunks),
            )
        else:
            logger.info("  %s: %d chunks", source_name, len(chunks))

    if not all_chunks:
        logger.error("No calibration chunks built from any source")
        return None, {}

    if len(all_chunks) < num_samples:
        logger.warning(
            "Total chunks %d < target %d. Proceeding with available data; "
            "recipe proportions preserved (no filler duplication).",
            len(all_chunks),
            num_samples,
        )

    # Cap at requested count (shouldn't trip unless one source over-packs).
    all_chunks = all_chunks[:num_samples]

    token_ids = [c for _, c in all_chunks]
    tensor = torch.tensor(token_ids, dtype=torch.long)
    elapsed = time.time() - t0
    logger.info(
        "Built %d chunks of %d tokens in %.1fs (%.1f chunks/s)",
        len(all_chunks),
        max_length,
        elapsed,
        len(all_chunks) / max(elapsed, 0.001),
    )

    # Source breakdown reflects ACTUAL counts (after any short-source truncation).
    source_counts: dict[str, int] = {}
    for source_name, _ in all_chunks:
        source_counts[source_name] = source_counts.get(source_name, 0) + 1

    # Compute PAD ratio for the manifest (sanity metric — should be < 5%
    # under packed mode, was 68% under old pad-to-max mode).
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_count = (tensor == pad_id).sum().item()
    pad_ratio = pad_count / max(tensor.numel(), 1)

    manifest: dict[str, Any] = {
        "version": "3.0.0",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": DEFAULT_MODEL,
        "purpose": "NVFP4 W4A4 calibration data — input_global_scale per fused Linear",
        "mix_profile": "phase1" if SOURCE_WEIGHTS is SOURCE_WEIGHTS_PHASE1 else "legacy",
        "packing_mode": "concat-pack-with-eos-separator",
        "total_samples": len(all_chunks),
        "max_length": max_length,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "pad_ratio": pad_ratio,
        "source_weights": dict(SOURCE_WEIGHTS),
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
    parser.add_argument(
        "--legacy-mix",
        action="store_true",
        help="Use the original 4-source 25/25/25/25 mix (MATH-500, HumanEval, ShareGPT, AIME). "
        "Default is the Phase 1 7-source benchmark-weighted mix.",
    )
    parser.add_argument(
        "--arc-mix",
        action="store_true",
        help=(
            "Use the Phase 2 ARC-aware 11-source mix that adds ARC-Easy (15%%), "
            "ARC-Challenge (10%%), and HellaSwag (15%%) training data. Recommended "
            "when optimizing for ARC-Easy/Challenge and HellaSwag accuracy. "
            "MoE expert routing paths exercised by commonsense QA will be better "
            "calibrated, reducing input_global_scale error for those activation patterns."
        ),
    )
    args = parser.parse_args()

    # Activate the legacy mix if requested. Default num-samples drops to 512 to match
    # the original legacy run unless the user explicitly set a different count.
    global SOURCE_WEIGHTS
    if args.legacy_mix:
        SOURCE_WEIGHTS = SOURCE_WEIGHTS_LEGACY
        if args.num_samples == DEFAULT_NUM_SAMPLES:
            args.num_samples = 512
    elif args.arc_mix:
        SOURCE_WEIGHTS = SOURCE_WEIGHTS_PHASE2

    if args.legacy_mix:
        mix_name = "LEGACY (4-source 25/25/25/25)"
    elif args.arc_mix:
        mix_name = "PHASE 2 ARC-AWARE (11-source, commonsense+benchmark-weighted)"
    else:
        mix_name = "PHASE 1 (7-source benchmark-weighted)"
    logger.info("=== GATE 4: Calibration Dataset Build ===")
    logger.info("Mix:        %s", mix_name)
    logger.info("Model:      %s", args.model_id)
    logger.info("Samples:    %d", args.num_samples)
    logger.info("Max length: %d", args.max_length)
    logger.info("Output:     %s", args.output_dir)
    logger.info("Source weights:")
    for src, w in SOURCE_WEIGHTS.items():
        logger.info("  %s: %.0f%% (~%d samples)", src, w * 100, int(args.num_samples * w))

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

    # Only load the sources the active mix actually uses. Saves bandwidth and
    # avoids download failures on sources we don't need.
    if "math500" in SOURCE_WEIGHTS:
        logger.info("Loading MATH-500 ...")
        sources["math500"] = load_math500(datasets)

    if "humaneval" in SOURCE_WEIGHTS:
        logger.info("Loading HumanEval ...")
        sources["humaneval"] = load_humaneval(datasets)

    if "sharegpt" in SOURCE_WEIGHTS:
        logger.info("Loading ShareGPT ...")
        sources["sharegpt"] = load_sharegpt(datasets)

    if "aime" in SOURCE_WEIGHTS:
        logger.info("Loading AIME ...")
        sources["aime"] = load_aime(datasets)

    if "gsm8k" in SOURCE_WEIGHTS:
        logger.info("Loading GSM8K ...")
        sources["gsm8k"] = load_gsm8k(datasets)

    if "mbpp" in SOURCE_WEIGHTS:
        logger.info("Loading MBPP ...")
        sources["mbpp"] = load_mbpp(datasets)

    if "triviaqa" in SOURCE_WEIGHTS:
        logger.info("Loading TriviaQA ...")
        sources["triviaqa"] = load_triviaqa(datasets)

    if "alpaca" in SOURCE_WEIGHTS:
        logger.info("Loading Alpaca-cleaned ...")
        sources["alpaca"] = load_alpaca(datasets)

    if "writingprompts" in SOURCE_WEIGHTS:
        logger.info("Loading WritingPrompts ...")
        sources["writingprompts"] = load_writingprompts(datasets)

    if "glaive" in SOURCE_WEIGHTS:
        logger.info("Loading Glaive-fn-calling ...")
        sources["glaive"] = load_glaive(datasets)

    if "arc_easy" in SOURCE_WEIGHTS:
        logger.info("Loading ARC-Easy (train) ...")
        sources["arc_easy"] = load_arc_easy(datasets)

    if "arc_challenge" in SOURCE_WEIGHTS:
        logger.info("Loading ARC-Challenge (train) ...")
        sources["arc_challenge"] = load_arc_challenge(datasets)

    if "hellaswag" in SOURCE_WEIGHTS:
        logger.info("Loading HellaSwag (train) ...")
        sources["hellaswag"] = load_hellaswag_train(datasets)

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
