# Decisions log

Running log of every non-obvious choice and why. Updated at every phase per
`PROJECT_SPEC.md` §6.

---

## Day 0 — scaffold

**Decision:** cloned the repo structure, phase-gated build plan, data
governance defaults, and doc set (`PROJECT_SPEC.md`, `ARCHITECTURE.md`,
`PLAN.md`) from `clinical-coding-eval`, generalizing the task-specific
content into an `OPEN_DECISIONS` block rather than guessing at the actual
task, data source, or sensitivity constraints.

**Why:** `clinical-coding-eval` already worked out a solid pattern for this
exact shape of project (small on-prem QLoRA model vs. large baseline,
under a hard data-locality constraint, evaluation-harness-first). The task
itself — what the company data is, what the model should learn, how
sensitive it is — was not specified, so guessing and building against a
fabricated schema would violate `PROJECT_SPEC.md` §6's "ask, don't assume"
rule. The `OPEN_DECISIONS` gate in §7 gets the same treatment the source
project gave GPU/OS/credentialing: a hard stop before Phase 1.

**Alternatives considered:** picking a plausible task (e.g. support-ticket
classification) and building it fully. Rejected — would produce a spec
that reads as if requirements were gathered when they were not, and any
code written against a guessed schema would likely need to be thrown away.

---

## Day 0 — OPEN_DECISIONS filled in

**Decision:** the user has no real proprietary company data to build this
against and asked for a recommendation. Picked **CUAD (Contract
Understanding Atticus Dataset)** as the data source, standing in for "our
own company data": 510 real commercial contracts, 41 expert-annotated
clause categories, each with a clause type + verbatim evidence span. Task
is extraction (label + evidence span per clause type present), matching
the `clinical-coding-eval` pattern this spec is cloned from. Full
`OPEN_DECISIONS` answers are in `PROJECT_SPEC.md` §7.

Compute: this machine has no GPU (checked via `nvidia-smi`, 31GB RAM,
Linux) — it runs the data pipeline, eval harness, and contamination probe.
The user has a separate GPU machine for Phase 5 (QLoRA training); exact
GPU model TBD, to be filled into `configs/train.yaml` before that phase.

**Why CUAD specifically:**
- Matches the repo's existing extraction + evidence-span shape without
  redesigning any of `src/eval/` (`grounding.py`, `schema.py` apply
  directly).
- Fixed, closed label space (41 categories) rather than one that needs to
  be empirically derived — simpler Phase 1, and §4.1 could be answered
  directly instead of left as "derive top-K."
- High-trust ground truth (structured legal-expert annotation process) —
  Phase 4 (gold annotation pass) can be skipped with a documented reason
  instead of requiring new annotation work.
- Being public data (not real confidential company data) makes Phase 3
  (contamination probe) a genuinely meaningful check rather than a
  formality to skip — a foundation model plausibly saw CUAD in
  pretraining, so proving the fine-tune adds signal beyond memorization is
  a real, checkable claim.
- Real enterprise-shaped task (legal/contract NLP) with a recognizable
  benchmark name — relevant for the "harness as the deliverable" framing
  in §0.

**Consequence:** because CUAD is public, §1.1's "never leaves the machine"
rule is *not* a real compliance requirement for this run — noted
explicitly in `OPEN_DECISIONS` #3 so it is never later misrepresented as
one. The pipeline still enforces it as a matter of practice, since
demonstrating that discipline correctly is part of what the harness is
meant to prove.

**Alternatives considered:** CFPB consumer complaint narratives (public,
real enterprise-adjacent classification data with a large label space) —
rejected in favor of CUAD because extraction + evidence span is a closer
match to the repo's existing eval modules and to the `clinical-coding-eval`
precedent this spec is cloned from.

**Still open:** exact large-model checkpoint for the baseline
(`OPEN_DECISIONS` #7) — deferred to build time per §6 ("verify checkpoint
names and library APIs at build time").

---

## Day 0 — training GPU confirmed; data source verified

**Decision:** user's training machine is 12GB VRAM / 32GB system RAM —
logged into `PROJECT_SPEC.md` `OPEN_DECISIONS` #4. This is materially
tighter than clinical-coding-eval's card the starting Phase 5 config was
sized for, so expect to reduce `max_seq_len` before rank/batch size if the
first training run OOMs, per the config's own note.

Verified the actual CUAD data source before writing any pipeline code
against a guessed schema (`PROJECT_SPEC.md` §6). `theatticusproject/cuad-qa`
on HuggingFace can't be loaded via `datasets.load_dataset` (arbitrary-code
loading scripts are no longer supported, and no auto-converted Parquet
exists for it). The official `theatticusproject/cuad` repo instead hosts
the raw CUAD v1 release directly: `CUAD_v1/CUAD_v1.json`, a SQuAD-style
file, downloaded and inspected directly.

**Confirmed schema (real, not assumed):**
- 510 contracts (`data[i].title`, one paragraph each — `context` is the
  full contract text).
- 41 `qas` per contract, one per clause type, matching CUAD's taxonomy
  exactly. Clause type is embedded in the question string
  (`related to "{clause_type}"`), not a separate field — the build script
  parses it out with a regex.
- Each qa has `is_impossible` + an `answers` list of `{text, answer_start}`
  — text is a verbatim substring of `context` (so grounding is
  checkable-by-construction), `answer_start` is a char offset. Multiple
  answers per qa are common (2,605 of 20,910 qas have >1 span — e.g. a
  contract citing "Governing Law" in two places) — `build_dataset.py` must
  emit one row per span, not assume a single answer.
- 6,702 positive (clause present) vs. 14,208 negative (`is_impossible`)
  qas — real class imbalance to report in `docs/data_report.md`, not an
  edge case to special-case away.

**Why this matters:** confirms `OPEN_DECISIONS` #2's description was
accurate (good — spec wasn't rewritten around a wrong guess), but the
*exact* record shape (clause type parsed from a question string, one row
per span not per qa) was not fully specified until this point and would
have been guessed wrong otherwise.

**Data source, for repro:**
`https://huggingface.co/datasets/theatticusproject/cuad/blob/main/CUAD_v1/CUAD_v1.json`

---

## Phase 1 — data pipeline complete

**Decision:** built `src/data/build_dataset.py` against the confirmed
schema above. One row per contract (not per clause-type or per span) —
a generation task needs the model to emit all clause types present in one
contract in a single pass, so grouping by contract is the natural unit,
not the raw per-`qa` shape CUAD ships in.

Dropped 1 exact-duplicate contract by content hash (`ADUROBIOTECH,INC...
CONSULTING AGREEMENT` vs. `...CONSULTING AGREEMENT(1)` — same text, two
titles) rather than by title, since title uniqueness doesn't guarantee
content uniqueness. 509 contracts remain.

Verified every evidence span is a verbatim substring of its contract's
context at *build time*, not just at eval time — this is a property of
CUAD's own data, so treating it as a raw-data invariant that fails loudly
if violated is stricter than deferring the check to `eval/grounding.py`
(which checks *model* output, a different thing entirely).

Split 80/10/10 by `contract_id`, seed 42 → 407 / 51 / 51 contracts.
`pytest tests/test_data.py` passes (10/10), including the no-leakage
assertion against the real parquet files (not just a synthetic check).
`docs/data_report.md` regenerates on every pipeline run — see it for the
full per-clause-type frequency table and span statistics (6,702 positive
clause instances, 2,605 of them with >1 span).

**Alternatives considered:** one row per (contract, clause_type) pair —
rejected because it would require N separate generations per contract at
inference time (41 calls instead of 1), which doesn't match how a
fine-tuned model would actually be used, and would make the schema/
grounding eval modules operate on the wrong unit.

**Acceptance test passed:** `pytest tests/test_data.py` (10/10, includes
no-leakage assertion) + `docs/data_report.md` exists with coverage,
distribution, and dedup counts, per `PROJECT_SPEC.md` §5 Phase 1.

---

## Task pivot — CUAD to NoteChat (2026-08-24)

**Decision:** the project's actual task is now **NoteChat clinical-note →
doctor-patient dialogue generation**, not CUAD contract-clause extraction.
`docs/PROJECT_SPEC.md` (§0, §4, §7) has been rewritten to match: the
original CUAD `OPEN_DECISIONS` answers are kept, marked `[SUPERSEDED]`,
for the historical record; a new §7a holds the current NoteChat answers.
`README.md` and `src/train/train.py` are being brought in line with the
notebook (`notebooks/finetune.ipynb`), which was already built against
NoteChat.

**Why this is being formally recorded now, not silently:** the pivot had
already happened in practice — `configs/data.yaml`, `src/data/build_dataset.py`,
`data/processed/*.parquet`, and `notebooks/finetune.ipynb` were all built
and run against NoteChat while `PROJECT_SPEC.md`, `docs/DECISIONS.md`, and
`src/train/train.py` still described CUAD. The notebook's own header cell
flagged this mismatch explicitly rather than hiding it. Per this repo's own
rule (§1.2: "never write a number that wasn't produced by a script," and
§6: "ask, don't assume") — leaving the docs pointing at a task the code no
longer performs is exactly the kind of silent drift `docs/DECISIONS.md`
exists to prevent. This entry, plus the `PROJECT_SPEC.md` rewrite, closes
that gap rather than leaving two authoritative-looking task descriptions
in the repo simultaneously.

**What carries over unchanged from the CUAD framing:** the phase-gated
build plan, the "harness is the deliverable" framing (§0), the local-only
data governance default (§1.1), and the four-arm baseline comparison
(§Phase 6) — none of that was CUAD-specific.

**What does not carry over:** `src/eval/grounding.py` and
`src/eval/schema.py` no longer apply — NoteChat's output is free-form
dialogue, not `{clause_type, evidence_span}` pairs, so there is no
verbatim-span grounding check and no fixed schema to validate. Phase 2
(eval harness) needs generation metrics (ROUGE/BERTScore against
`conversation`) instead, carrying the ground-truth caveat from
`PROJECT_SPEC.md` §4.2 every time it's reported. `configs/eval.yaml`'s
`schema.enabled: false` already anticipated this.

**Real numbers on disk today** (superseding the CUAD counts in the "Phase 1
— data pipeline complete" entry above, which described a run against CUAD
that predates this pivot): 10,000 NoteChat notes, 0 exact-duplicates
dropped, split 8,000 / 1,000 / 1,000 by `note_id`, seed 42. Full stats in
`docs/data_report.md` (auto-generated, current).

**Alternatives considered:** reverting the code back to CUAD to match the
existing spec instead of rewriting the spec to match the code. Rejected —
the CUAD data pipeline no longer exists in this repo (no CUAD raw file,
no `label_space_from`-independent build path), NoteChat data is already
processed and a training notebook already runs against it, and discarding
that working state to chase the original spec would be pure rework with
no upside now that the user has confirmed NoteChat as the real task.

**Still open:** `src/train/train.py` (the CLI trainer) still contains the
CUAD-specific system prompt, `label_space_from`, and JSON-clause target
builder — being rewritten to mirror the notebook's NoteChat prompt/target
functions in the same change that added this entry, so the CLI trainer and
the notebook don't drift apart on task definition the way the docs did.

---

## Phase 5 — QLoRA fine-tune complete (2026-08-25)

**Decision:** the real training run described in
`docs/finetune_kernel_setup_status.md` (started, then deliberately stopped
to disconnect the external drive) was re-run to completion on the RTX 3060
machine via `notebooks/finetune.ipynb`. `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`,
LoRA r=16 (29.9M / 3.12B params, 0.96%), 2 epochs, 250 steps, effective
batch 16, `max_seq_len=4096`.

**Real numbers from `trainer_state.json`:** train loss 1.229 → 1.081 →
1.045 (steps 50/100/250); `eval_loss` 1.063 at the epoch-1 boundary
(step 125). No epoch-2 eval entry is present in `log_history` — checkpoints
at steps 150/200/250 and `final_adapter/` (saved 2026-08-25 08:04) all
exist on disk, so training itself ran to completion; the missing second
eval needs checking before trusting an epoch-2 number in any report (could
be `eval_strategy` config, an interrupted eval pass, or intentional —
confirm before Phase 2 references it).

**Sanity-check generation (2026-08-25):** ran the notebook's inference logic
against `final_adapter` on 3 val examples via a standalone script (loading
the saved adapter fresh rather than re-running the notebook from cell 1,
which would retrain). Output reliably matches the Doctor:/Patient: turn
format and stays grounded in the clinical note's stated facts (vitals,
findings, procedures) — structurally close to the NoteChat reference style.
One real quality issue found: sample 0's generation appended an unprompted,
fabricated aside — `(If the patient eventually dies) Doctor: I'm sorry to
inform you...` — content with no basis in the source note or reference.
Flagging as a hallucination/faithfulness failure mode to check for
systematically in the real eval harness, not just via ROUGE/BERTScore
similarity.

**Still open:** Phase 2's actual eval harness (ROUGE/BERTScore against
`conversation`, replacing the CUAD-specific `grounding.py`/`schema.py`, plus
a hallucination check per the finding above) has not been built yet; per
`docs/PLAN.md` Day 5 this is the next acceptance gate, not the training run
itself.

---

## Phase 2 — eval harness built (2026-08-25)

**Decision:** built the harness out of order relative to §5's "before any
training" sequencing (Phase 5 already ran) — the training run happened
first in practice, so this phase now validates against both the zero-shot
and fine-tuned arms in the same pass rather than zero-shot-only. Noting the
out-of-order build explicitly rather than silently pretending the intended
sequencing was followed.

`src/eval/schema.py` and `src/eval/grounding.py` were never created (§4
already established they don't apply to free-form dialogue output — nothing
to delete). Built instead:

- `src/inference/local_hf.py` — `SYSTEM_PROMPT`/`build_user_prompt`/
  `load_model`/`generate`, factored out of `notebooks/finetune.ipynb`'s
  inference cells so `run_eval.py` doesn't duplicate that logic in a second
  place that can drift. `load_model(adapter_path=None)` loads the frozen
  base model (zero-shot arm); passing `adapter_path` loads a saved LoRA
  adapter on top — same code path serves Phase 6 arms #1 and #3.
- `src/eval/metrics.py` — `rouge` (rouge1/rouge2/rougeL F-measure),
  `bertscore` (batched P/R/F1 via `bert-score`, lazily imported since it
  loads a roberta-large scorer), `turn_format_validity` (structural check:
  does the raw generation parse as Doctor:/Patient: turns — the closest
  analogue this task has to `schema.py`'s validity-rate metric, not a
  strict-alternation check since real conversations don't strictly
  alternate per `docs/data_report.md`), `bootstrap_ci` (1000-resample
  percentile CI at the record level), `paired_bootstrap_delta` (CI on the
  mean paired A-vs-B difference, for the Phase 6 baseline comparisons).
  Added `rouge-score` and `bert-score` to `pyproject.toml`'s base
  dependencies (not the `gpu` extra — the eval harness should still run
  metric-only on the non-GPU machine per `OPEN_DECISIONS` #4's original
  split, even though in practice everything currently runs on the one GPU
  machine).
- `src/eval/run_eval.py` — single entrypoint (`python -m src.eval.run_eval
  --arm <name> [--adapter <path>]`). Samples `configs/eval.yaml`'s
  `val_records` (200) from `data/processed/test.parquet` (seeded), runs
  generation, computes per-record + bootstrapped aggregate metrics plus
  tokens/sec, wall-clock, and peak VRAM (`torch.cuda.max_memory_allocated`),
  writes `artifacts/eval/{arm}/results.json`.

**Unit tests** (`tests/test_eval.py`, 12/12 passing): hand-computed toy
examples for `rouge` (identical/disjoint/partial-overlap strings),
`turn_format_validity`, `bootstrap_ci` (constant-value zero-width CI,
determinism given a seed), and `paired_bootstrap_delta` (identical arrays →
zero delta, constant-offset arrays → exact delta). `bertscore` has no unit
test — it needs a downloaded roberta-large scorer, so it's exercised via
`run_eval.py`'s end-to-end run instead, not a fast unit test.

**Smoke-tested** end-to-end on 3 zero-shot records before committing to a
full run: ROUGE/BERTScore/format-validity/latency/VRAM all populated
correctly in the output JSON (deleted after confirming, not part of the
real result set).

**Full run launched:** `--arm zero-shot` then `--arm finetuned` (adapter =
`artifacts/adapters/unsloth__Qwen2.5-3B-Instruct-bnb-4bit/final_adapter`),
200 records each, sequentially (same 12GB card, can't run both at once).
Numbers to be logged here once both `results.json` files exist — per §1.2,
not written into this entry ahead of the run actually producing them.

**Fixed:** `tests/test_data.py::test_load_raw_rejects_unexpected_columns`
hardcoded `/tmp/_bad_schema_test.csv`, which doesn't exist on this Windows
machine outside WSL — replaced with pytest's `tmp_path` fixture (a real
per-test temp dir on any OS). `pytest tests/test_data.py` is 12/12 now.

**Sequencing risk, checked rather than just flagged:** the ordering
violation above (harness built after Phase 5's training run, not before)
matters only insofar as it could let the metric choice get unconsciously
tuned to flatter the model — that's the actual failure mode §5 Phase 2's
"metrics before models" rule exists to prevent, not the calendar order for
its own sake. Checked whether that happened here: the implemented metric
set (`rouge`, `bertscore`, `turn_format_validity`, `bootstrap_ci`,
`paired_bootstrap_delta`) is exactly PROJECT_SPEC.md §5 Phase 2's
prescribed table, chosen before this entry was written and unchanged by
anything observed in the trained model's output. The one thing that *was*
informed by having already seen the fine-tuned model's generations (the
sanity-check pass earlier today, which surfaced a fabricated aside not
grounded in the source note) is a hallucination/faithfulness check — and
that was explicitly *not* implemented in this pass, only recorded as future
work above. So the out-of-order build left a documented gap (no
hallucination metric yet) but did not bias what *was* measured.
