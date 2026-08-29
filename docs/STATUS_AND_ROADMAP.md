# Status & Roadmap — as of 2026-08-28

Snapshot of what's actually built and what a next iteration would tackle.
Companion to `PROJECT_SPEC.md` (the spec) and `DECISIONS.md` (the why) — this
file is the "where things stand" view. It goes stale fast; treat it as a
snapshot, not a live doc, and trust `artifacts/eval/` over any prose here.

---

## 1. What's done

All seven phases are complete. Every arm was scored on the same 200 held-out
records under greedy decoding.

| Phase | Status | Evidence |
|---|---|---|
| Task selection + spec | Done, pivoted once | `PROJECT_SPEC.md` §7a — NoteChat clinical-note → dialogue generation, superseding an earlier CUAD spec (`DECISIONS.md` "Task pivot") |
| Phase 1 — Data pipeline | **Done** | `src/data/build_dataset.py`; 10,000 notes → 8,000/1,000/1,000 split by `note_id`, seed 42, 0 duplicates; `docs/data_report.md` |
| Phase 2 — Eval harness | **Done** | `src/eval/{metrics,faithfulness,run_eval,compare}.py`; bootstrap CIs, paired deltas, numeric-grounding proxy, turn-format check |
| Phase 3 — Contamination probe | **Done** | `src/eval/contamination.py` → `docs/contamination_report.md`; prefix continuation on the base model against a **deranged** control. Finding: memorization is **detectable but small** (+0.021 ROUGE-L, CI excludes zero) — real, and ~10× too small to explain the fine-tuning gains |
| Phase 4 — Gold annotation | Skipped, documented | `PROJECT_SPEC.md` §7a item 6 — a reasoned skip (the reference is LLM-generated; a human "gold" dialogue wouldn't fix that), not an oversight |
| Phase 5 — QLoRA fine-tune | **Done** | `notebooks/finetune.ipynb`; Qwen2.5-3B-Instruct-bnb-4bit, LoRA r=16, 2 epochs / 250 steps; adapter at `artifacts/adapters/.../final_adapter` |
| Phase 6 — Baselines (4 arms) | **Done** | Zero-shot 3B, zero-shot 14B, QLoRA 3B, TF-IDF retrieval. **Headline: the fine-tuned 3B beats the zero-shot 14B on every content metric, all 95% CIs excluding zero** — and the no-model retrieval baseline beats the 14B on ROUGE-1 while fabricating 68pp more numbers |
| Phase 7 — Write-up | **Done** | `README.md`, `docs/MODEL_CARD.md`, `docs/DECISIONS.md` |
| Tests + CI | **Done** | 64 tests, GitHub Actions on every push/PR, CPU-only (no CUDA in CI) |

---

## 2. The findings, in one paragraph

QLoRA fine-tuning a 3B model bought **more than double** what a ~5×
parameter increase bought zero-shot (+0.346 vs. +0.153 ROUGE-1), while
running faster in ~4× less VRAM. But the TF-IDF retrieval baseline — no
model, no training, no GPU — **also beat the 14B model on ROUGE-1**, purely
by returning a fluent, on-style dialogue about a different patient. That
result is why `faithfulness.py` exists: on this task, reference-similarity
metrics substantially reward style- and topic-matching rather than
faithfulness, and the numeric-grounding metric is what separates the arms
that actually read the note from the one that doesn't.

The contamination probe found a small but real memorization signal in the
base model (+0.021 ROUGE-L over a deranged control, CI excluding zero). It is
roughly an order of magnitude smaller than the fine-tuning effect, so it does
not explain the result — but the repo says "small and real", not "clean".

---

## 3. What a next iteration would tackle

Ranked by how much it would change a reader's conclusions, not by effort.

### Would change conclusions
1. **Ablations** (LoRA rank 8/16/32, 1 vs. 2 epochs). `PLAN.md` and the spec
   call these out explicitly. Only the single default config has been run,
   so "r=16, 2 epochs" is currently an unjustified choice rather than a
   measured one — the one place this project asserts a hyperparameter
   without evidence behind it.
2. **A faithfulness metric that catches non-numeric fabrication.**
   `faithfulness.py` measures numeric grounding only, so a fabricated
   *diagnosis* stated without a number is invisible to it. This is stated as
   a limitation in the model card, but it is the largest remaining gap
   between what the harness measures and what "faithful" means.
3. **Score all 1,000 test records instead of 200.** Deliberately deferred —
   `configs/eval.yaml` justifies the subsample quantitatively (CI
   half-widths are 25-40× smaller than the effects) — but it would be
   required for any comparison with a genuinely small effect size, e.g. the
   ablations in item 1.

### Documentation debt
4. **The missing epoch-2 `eval_loss`** (`DECISIONS.md`'s Phase 5 entry) is
   still unrecoverable for the committed adapter. `src/train/train.py` now
   calls `trainer.evaluate()` explicitly so a future run always records it,
   but the existing number is gone without retraining.
5. **`tokens_per_second` in the committed results predates a counting fix.**
   See `DECISIONS.md` — the figures are from re-tokenized decoded text; the
   code now counts generated ids directly. A full rerun would shift them
   slightly (the counts are the only affected field; no quality metric or
   delta changes).

### Not needed
Per `PROJECT_SPEC.md` §8: no web UI, no API server, no Docker, no RAG.
Adding any of these would work against the project's own framing — it is
judged on measurement rigor, not deployment surface area.

---

## 4. Added outside the phase structure

**Voice input (2026-08-29).** `scripts/voice_to_note.py` — local
speech-to-text (faster-whisper) feeding the same `try_model.py` code path,
so a spoken note is scored by the same checks as a typed one. `uv sync
--extra gpu --extra voice`. Not a phase or a research result — a CLI
convenience, and explicitly not a web UI or API server (`DECISIONS.md`,
"Voice input added").
