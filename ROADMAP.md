# Roadmap

## Vision

ZAYA1-8B is a state-of-the-art reasoning model with 760M active parameters that
outperforms models 5x its size on math and coding. Per its own technical report
(arXiv 2605.05365, May 2026), Zyphra **deliberately skipped** the multi-turn
agentic RL stage — the agentic gap is a training omission, not a capability ceiling:

> "ZAYA1-8B does not include a dedicated multi-turn agentic RL stage in this
> release. We include some supervised agent, tool, and SWE traces during SFT,
> but the RL cascade is primarily optimized for verifiable reasoning, math,
> code, and instruction-following behavior. Scaling agentic data and agentic
> RL is left for future releases."

**This project completes ZAYA1-8B**: adding the missing agentic stage to drive
the Godspeed coding agent on consumer hardware (RTX 5070 Ti, 16 GB VRAM).

## Status Legend

| Icon | Meaning |
|------|---------|
| ✅ | Done |
| 🔴 | Blocked |
| 🟡 | In progress |
| ⬜ | Not started |

---

## Phase 1 — Compatibility Gate ✅

**Goal**: Verify the Zyphra stack can load ZAYA1-8B and attach PEFT adapters.

| Task | Status | Notes |
|------|--------|-------|
| Install Zyphra transformers fork | ✅ | `transformers @ git+https://github.com/Zyphra/transformers.git@zaya1` |
| Load ZAYA1-8B config and verify architecture | ✅ | `ZayaForCausalLM`, 80 layers (40 attn + 40 MoE), CCA |
| Map all 1641 Linear modules for LoRA targeting | ✅ | 200 attention, 1280 expert, 160 router, 1 lm_head |
| Attach PEFT LoRA to attention projections | ✅ | 8.2M trainable params, 0.09%, 7.24 GB VRAM |
| Verify gradient flow through LoRA adapters | ✅ | 400 params with gradients, loss.backward() succeeds |
| Document architecture and compatibility | ✅ | `COMPATIBILITY.md` |

**Decision**: Gate PASSED. QLoRA fine-tuning is viable.

---

## Phase 2 — Inference Pipeline 🔴

**Goal**: Serve ZAYA1-8B via an OpenAI-compatible endpoint at usable speed.

| Task | Status | Notes |
|------|--------|-------|
| Build vLLM (Zyphra fork) in WSL | ✅ | CUDA 13.2 toolkit, 64 GB WSL, MAX_JOBS=10. 65-minute build. `scripts/build_vllm_detached.sh`. |
| Serve ZAYA1-8B via vLLM | 🔴 | Blocked: Windows desktop compositor consumes ~15.9 GB VRAM. 85 MB free. Needs cold reboot. |
| Verify generation quality (coding prompts) | ⬜ | Test with Godspeed-style system prompts |
| Benchmark throughput on RTX 5070 Ti (16 GB) | ⬜ | Target: 50+ tok/s via vLLM |
| Fix serve_zaya1.py in Godspeed repo | ✅ | `scripts/serve_zaya1.py` with n-gram speculation + tool-call support |
| NF4 path (transformers) | ❌ | Confirmed broken — bitsandbytes dequant incompatible with CCA attention |

**Blocker**: Windows desktop compositor uses ~15.9 GB VRAM. Cold reboot frees 14+ GB.
**Alternative (interim)**: Zyphra Cloud API at cloud.zyphra.com — no local VRAM needed.

---

## Phase 3 — Teacher Trajectory Generation 🟡

**Goal**: Generate 300–500 mechanically verified tool-calling trajectories using
DeepSeek V4 Pro via NVIDIA NIM as teacher, routed through Godspeed.

**Teacher**: `nvidia_nim/deepseek-ai/deepseek-v4-pro`
- Released April 24, 2026 | MIT license | 1.6T total / 49B active MoE
- SWE-bench Verified: 80.6% | LiveCodeBench: 93.5% | Codeforces: 3,206
- Available on NVIDIA NIM free tier (4 keys available)
- Rate limits: ~30 RPM per key → ~120 RPM effective with rotation → ~13 min for 200 tasks

| Task | Status | Notes |
|------|--------|-------|
| Configure Godspeed for NIM DeepSeek V4 Pro | ⬜ | `settings.yaml` with model routing: plan via v4-pro-max, edit via v4-pro |
| Run Godspeed headless against 20-task suite | ⬜ | Baseline trajectory quality check |
| Expand 20 base tasks to 200+ variants | ✅ | `scripts/mutate_tasks.py` — 6 mutation types + 10 OOD tasks |
| Run Godspeed headless against 200 mutated tasks | ⬜ | Each task runs in headless mode, conversation logger captures JSONL |
| Filter trajectories through quality gates | ⬜ | `scripts/remap_to_zaya.py` — exit code 0, no dangerous commands, schema valid |
| Remap to ZAYA XML ChatML format | 🟡 | `scripts/remap_to_zaya.py` built — confirmed 0/1325 old sessions pass (expected) |
| Validate tool coverage (30+ Godspeed tools) | ⬜ | |
| Split train/val/eval (80/10/10) | ⬜ | |

**Quality gates** (mandatory, per context doc):
1. Mechanical verify hook passed (13/20 tasks have hooks)
2. Jaccard tool selection >= 0.7 vs expected tools
3. Zero dangerous command flags in reward annotations
4. Zero schema validation errors in tool call sequence
5. Session exit code 0 (success) from headless mode

**Volume target**: 300–500 verified trajectories. ZAYA's MoE (760M active) has high
sample efficiency. Verified 300 >> unverified 3,000. Do not scale volume before
fixing quality.

---

## Phase 4 — ZAYA XML Format & Tool Schema ⬜

**Goal**: Ensure training data uses ZAYA1-8B's exact tool-call format.

ZAYA1-8B uses a JSON-inside-XML format via vLLM's `--tool-call-parser zaya_xml`:
```
<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
```

This differs from Godspeed's native Qwen3-Coder XML format (`<function=name>`) and
from OpenAI-standard `tool_calls` JSON. A remapper is required BEFORE data enters
Unsloth/TRL — silent corruption otherwise.

| Task | Status | Notes |
|------|--------|-------|
| Build ZAYA XML format remapper | ✅ | `scripts/remap_to_zaya.py` — Godspeed JSONL → ZAYA ChatML |
| Verify ZAYA chat template compatibility | ⬜ | `tokenizer.apply_chat_template()` with `<|im_start|>` / `<|im_end|>` tokens |
| Spot-check 50 remapped trajectories manually | ⬜ | Before any training run — this is the highest-risk item |
| Test vLLM `zaya_xml` parser with sample output | ⬜ | Send remapped data through vLLM, verify tool calls parse |

---

## Phase 5 — QLoRA Fine-Tuning 🟡

**Goal**: Train ZAYA1-8B to produce valid tool calls in Godspeed's XML format.

**Strategy**: SFT → Rejection Sampling → GRPO (2026 SOTA pipeline from published models)

| Task | Status | Notes |
|------|--------|-------|
| Training script with TRL SFTTrainer | ✅ | `scripts/train.py` — QLoRA with config-driven pipeline, dry-run support |
| Dry run (1 batch, no save) to catch OOM | ⬜ | Target: <12 GB VRAM |
| SFT Stage 1: 1–2 epochs on 100 verified trajectories | ⬜ | Format learning only. Monitor AIME regression (<5% tolerance) |
| Evaluate baseline checkpoint on 20-task suite | ⬜ | Primary metric: Jaccard + mech verify |
| Full SFT on 300–500 verified trajectories | ⬜ | `WANDB_MODE=offline` on Windows |
| Save LoRA adapter weights | ⬜ | |
| GRPO Stage 2: policy improvement via verifiable rewards | ⬜ | 4–8 rollouts per prompt, mechanical verify as primary reward |
| Merge adapter (optional, for vLLM deployment) | ⬜ | |

**VRAM Budget (QLoRA, 16 GB GPU)**:

| Component | VRAM |
|-----------|------|
| NF4 base model | 7.2 GB |
| LoRA adapters | ~0.0 GB |
| Optimizer (8-bit Adam) | 0.3 GB |
| Gradients | 0.3 GB |
| Activations (w/ gradient checkpointing) | 2–4 GB |
| **Total** | **10–12 GB ✓** |

**Catastrophic forgetting mitigation**: QLoRA, conservative rank (r=16), 1–2 epochs max.
Hard stop if AIME degrades >5% from baseline (89.1%).

---

## Phase 6 — Evaluation ⬜

**Goal**: Measure tool-calling accuracy improvement and close the agentic gap.

| Task | Status | Notes |
|------|--------|-------|
| Godspeed 20-task internal benchmark (Jaccard + mech verify) | ⬜ | Primary iteration metric — fast enough for every checkpoint |
| BFCL-v4 | ⬜ | Current baseline: 39.22 vs Qwen3-4B 49.7 (+10pt gap to close) |
| τ² (agentic) | ⬜ | Current baseline: 43.12 vs Qwen3.5-4B 82.1 (+39pt gap) |
| SWE-bench dev-23 subset | ⬜ | Godspeed has existing infra from benchmark runs |
| Regression: AIME subset + LiveCodeBench subset | ⬜ | Catch catastrophic forgetting |
| Compare against Qwen2.5-Coder-14B baseline | ⬜ | Current Godspeed default driver |
| Ablation: LoRA rank (r=8 vs r=16 vs r=32) | ⬜ | Optional |
| Ablation: with/without router layers in targets | ⬜ | Optional |

**Target outcomes**:
- Conservative (300 trajectories, SFT only): BFCL-v4 ~44–47, τ² ~55–65
- Aggressive (500 trajectories, SFT + GRPO): BFCL-v4 ~50+, τ² ~65–75

Do NOT expect τ² parity with Qwen3.5-4B (82.1) in one cycle — that model has
dedicated agentic RL at scale. This is a first cycle.

---

## Phase 7 — Production Deployment ⬜

**Goal**: Merge fine-tuned ZAYA1-8B into Godspeed as a first-class driver.

| Task | Status | Notes |
|------|--------|-------|
| Merge adapter or export full model | ⬜ | |
| Add to Godspeed model presets | ⬜ | `--preset zaya` already exists |
| Update driver catalog with fine-tuned benchmark scores | ⬜ | |
| Write model card on HuggingFace | ⬜ | Community contribution |
| Publish findings (blog post / technical note) | ⬜ | Portfolio piece |

---

## Pipeline Architecture

```
mutated_tasks.jsonl (200 tasks)
       │
       ▼
Godspeed headless ─── DeepSeek V4 Pro (NIM)
       │
       ▼
~/.godspeed/training/*.conversation.jsonl
       │
       ▼
remap_to_zaya.py (quality gates + ZAYA XML format)
       │
       ▼
data/train_zaya.jsonl (300–500 verified ChatML trajectories)
       │
       ▼
train.py ─── Unsloth + TRL SFTTrainer (QLoRA, r=16)
       │
       ▼
checkpoints/zaya1-tool-call/
       │
       ▼
vLLM serve (evaluation) → Godspeed 20-task benchmark → BFCL-v4
```

## Known Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Catastrophic forgetting (AIME 89.1 → <84) | Loss of ZAYA's core reasoning | QLoRA r=16, low LR, 1–2 epochs. Monitor AIME every epoch. Hard stop if >5%. |
| Tool call schema mismatch | Model trains on garbage silently | Manual spot-check 50 rows after remapping. Test through vLLM zaya_xml parser. |
| Shopify OOD failure mode | Good benchmarks, bad real usage | 30% OOD tasks minimum. Mutate before generation, not after. |
| Teacher format hallucination | DeepSeek V4 generates invalid tool calls | Godspeed's schema validator catches these at runtime. Filter during remapping. |
| Benchmark leakage | SWE-bench tasks overlap with training data | Cross-check tasks.jsonl against SWE-bench Verified task IDs before release eval. |
| No GGUF / llama.cpp support | Can't use spec decoding or local llama.cpp server | vLLM is the only viable local path |
| Zyphra fork required | Extra build step for both transformers and vLLM | One-time setup, documented |
| 16 GB VRAM ceiling | ~24K context max, limited batch size | Adequate for agent loop (single sequence) |

## Reference Links

- [Zyphra/ZAYA1-8B on HuggingFace](https://huggingface.co/Zyphra/ZAYA1-8B)
- [Technical Report (arXiv 2605.05365)](https://arxiv.org/abs/2605.05365)
- [Zyphra Blog Post](https://www.zyphra.com/post/zaya1-8b)
- [Zyphra vLLM Fork](https://github.com/Zyphra/vllm/tree/zaya1-pr)
- [Zyphra Transformers Fork](https://github.com/Zyphra/transformers/tree/zaya1)
- [NVIDIA NIM — DeepSeek V4 Pro](https://build.nvidia.com/deepseek-ai/deepseek-v4-pro)
- [Godspeed Coding Agent](https://github.com/omnipotence-eth/godspeed-coding-agent)
