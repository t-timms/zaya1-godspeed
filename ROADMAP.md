# Roadmap

## Vision

ZAYA1-8B is a state-of-the-art reasoning model with 760M active parameters that
outperforms models 5x its size on math and coding. It was not trained for
multi-step tool calling — the interaction pattern Godspeed's agent loop depends on.

**This project produces a fine-tuned ZAYA1-8B variant that drives the Godspeed
coding agent reliably**, matching or exceeding the tool-calling accuracy of
the current Qwen2.5-Coder-14B default while using less VRAM and running faster.

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

## Phase 2 — Inference Pipeline 🟡

**Goal**: Serve ZAYA1-8B via an OpenAI-compatible endpoint at usable speed.

| Task | Status | Notes |
|------|--------|-------|
| Build vLLM (Zyphra fork) in WSL | ✅ | CUDA 13.2 toolkit, 64 GB WSL, MAX_JOBS=10. 65-minute build. `scripts/build_vllm_detached.sh`. |
| Serve ZAYA1-8B via vLLM | 🟡 | Architecture resolved correctly (ZayaForCausalLM). Blocked: Windows desktop VRAM (12GB). Needs reboot or Cloud API. |
| Verify generation quality (coding prompts) | ⬜ | Test with Godspeed-style system prompts |
| Benchmark throughput on RTX 5070 Ti (16 GB) | ⬜ | Target: 50+ tok/s via vLLM |
| Fix serve_zaya1.py in Godspeed repo | ✅ | `scripts/serve_zaya1.py` with n-gram speculation + tool-call support |
| NF4 path (transformers) | ❌ | Confirmed broken — bitsandbytes dequant incompatible with CCA attention |

**Blocker**: Windows desktop compositor uses 12 GB VRAM. Cold reboot frees 14+ GB, enough for BF16 model loading.
**Blocker**: Windows desktop compositor uses 12 GB VRAM. Cold reboot frees 14+ GB, enough for BF16 model loading.

**Alternative (interim)**: Zyphra Cloud API at cloud.zyphra.com — no local VRAM needed.

---

## Phase 3 — Godspeed Integration ⬜

**Goal**: ZAYA1-8B drives Godspeed's agent loop end-to-end.

| Task | Status | Notes |
|------|--------|-------|
| Verify LiteLLM `openai/zaya1-8b` routing works | ⬜ | Point `OPENAI_BASE_URL` at vLLM or Zyphra Cloud |
| Run `validate_driver.py` smoke (3 SWE-Bench Lite instances) | ⬜ | Gate: must produce patch on ≥1 instance with <20% LLM errors |
| Test tool calling: read, write, shell, grep, glob | ⬜ | Core Godspeed tool set |
| Test permission engine interaction | ⬜ | Deny-first, 4-tier gating must work |
| Benchmark agent loop latency vs Qwen2.5-Coder-14B | ⬜ | Current baseline: ~78 tok/s (750 with spec dec) |
| Update Godspeed driver catalog | ⬜ | Add vLLM notes, update context window, benchmark scores |

---

## Phase 4 — Training Data Generation 🟡

**Goal**: Produce 500–2K high-quality tool-calling trajectories for fine-tuning.

| Task | Status | Notes |
|------|--------|-------|
| Data pipeline script | ✅ | `data/generate.py` — extracts trajectories from Godspeed JSONL, filters quality, outputs ChatML |
| Run Godspeed benchmark suite with strong API model | ⬜ | Claude/GPT as actor to generate clean trajectories |
| Export sessions from Godspeed audit trail | ⬜ | JSONL with per-step reward annotations |
| Filter for successful tool-call sequences | ⬜ | No rejected/retried tool calls in trajectory |
| Convert to ChatML format | ✅ | Auto-handled by `data/generate.py` |
| Validate schema coverage (all 30+ Godspeed tools represented) | ⬜ | |
| Split train/val/eval | ⬜ | 80/10/10 split |

---

## Phase 5 — QLoRA Fine-Tuning 🟡

**Goal**: Train ZAYA1-8B to produce valid tool calls in Godspeed's XML/JSON format.

| Task | Status | Notes |
|------|--------|-------|
| Training script with TRL SFTTrainer | ✅ | `scripts/train.py` — QLoRA with config-driven pipeline, dry-run support |
| Dry run (1 batch, no save) to catch OOM | ⬜ | Target: <12 GB VRAM |
| Full training run (3 epochs, ~500-2K examples) | ⬜ | `WANDB_MODE=offline` on Windows |
| Save LoRA adapter weights | ⬜ | |
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

---

## Phase 6 — Evaluation ⬜

**Goal**: Measure tool-calling accuracy improvement over base ZAYA1-8B.

| Task | Status | Notes |
|------|--------|-------|
| Tool-call schema validation pass rate | ⬜ | Primary metric — the failure mode we're fixing |
| `validate_driver.py` smoke (same 3 instances) | ⬜ | Compare pre- and post-fine-tune |
| Full 20-task Godspeed benchmark | ⬜ | |
| Compare against Qwen2.5-Coder-14B baseline | ⬜ | Not necessarily beating it, but closing the gap |
| Ablation: LoRA rank (r=8 vs r=16 vs r=32) | ⬜ | Optional |
| Ablation: with/without router layers in targets | ⬜ | Optional |

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

## Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| No GGUF / llama.cpp support | Can't use spec decoding or local llama.cpp server | vLLM is the only viable local path |
| Zyphra fork required | Extra build step for both transformers and vLLM | One-time setup, documented |
| 16 GB VRAM ceiling | ~24K context max, limited batch size | Adequate for agent loop (single sequence) |
| No Unsloth support | Can't use Unsloth's memory optimizations | TRL + bitsandbytes is sufficient |
| MoE architecture | Expert routing may interfere with tool-call formatting | Target attention projections only, not expert weights |

## Reference Links

- [Zyphra/ZAYA1-8B on HuggingFace](https://huggingface.co/Zyphra/ZAYA1-8B)
- [Technical Report](https://arxiv.org/abs/2605.05365)
- [Zyphra vLLM Fork](https://github.com/Zyphra/vllm/tree/zaya1-pr)
- [Zyphra Transformers Fork](https://github.com/Zyphra/transformers/tree/zaya1)
- [llama.cpp Feature Request #22776](https://github.com/ggml-org/llama.cpp/issues/22776)
- [Godspeed Coding Agent](https://github.com/omnipotence-eth/godspeed-coding-agent)
