# Execution Plan — On-Prem Fine-Tuning on Company Data

Companion to `PROJECT_SPEC.md`. Read that first — this file is the
calendar, not the spec. Cloned from `clinical-coding-eval`'s `PLAN.md`
structure; the day-by-day content below is generic until `OPEN_DECISIONS`
is filled in.

---

## 0. Reality check (read before anything else)

`PROJECT_SPEC.md` §7 `OPEN_DECISIONS` is not filled in yet. That block —
not code — is the actual critical path for this project. Nothing in Phase 1
can start until the task, the data location, and the sensitivity
constraints are answered, because every downstream choice (model size,
label space, split key, whether the large-baseline arm can call an
external API at all) depends on those answers.

**Implication:** the plan below is written so that once `OPEN_DECISIONS`
is answered, Day 1 (repo scaffold — already done here) rolls straight into
Day 2 (eval harness, which can often be built and unit-tested against
synthetic fixtures before real data access is sorted out).

---

## 1. What to do right now

1. Fill in `PROJECT_SPEC.md` §7 `OPEN_DECISIONS` — task type, data
   description, sensitivity constraints, compute, split key, ground-truth
   status, baseline model.
2. Once that's answered, rewrite `PROJECT_SPEC.md` §0, §4, and
   `ARCHITECTURE.md`'s diagrams around the real task — they are
   placeholders right now, not a spec to build against.
3. Say "go" and Day 1 proper (environment + first real config values)
   starts.

---

## 2. Day-by-day (template — adjust once `OPEN_DECISIONS` is answered)

### Day 1 — Environment + config
1. Fill `pyproject.toml` dependencies for the actual model/library choices.
2. `uv venv` + `uv sync`; verify the training stack imports cleanly on the
   actual training machine, not just this one.
3. Fill `configs/data.yaml`, `configs/train.yaml`, `configs/eval.yaml` with
   real values instead of `null`/`TBD`.
4. Start `docs/DECISIONS.md` — log every choice made today with rationale.

**Exit test:** the core dependency imports needed for Phase 1 run clean.

### Day 2 — Eval harness core (Phase 2 groundwork, data-independent where possible)
1. `src/eval/metrics.py` — the task metric(s), with bootstrap CI. Unit
   tests with hand-computed expected values.
2. Any of `schema.py` / `grounding.py` / `calibration.py` / `selective.py`
   that apply to this task (see `PROJECT_SPEC.md` §5 Phase 2).
3. Build a small, clearly-synthetic fixture set to exercise the harness
   end to end before it ever touches real company data.
4. `src/eval/run_eval.py` skeleton wired to the synthetic fixtures.

**Exit test:** `pytest tests/test_eval.py` passes; `run_eval.py` runs end
to end on synthetic data with every result field populated.

### Day 3 — Data pipeline (Phase 1, real data)
1. `src/data/build_dataset.py` against the real source named in
   `OPEN_DECISIONS` #2.
2. Derive the label/output space empirically (§4.1) — report it in
   `docs/data_report.md`, do not hardcode from assumption.
3. Split on the correct key (`OPEN_DECISIONS` #5); assert zero overlap in
   a test.

**Exit test:** `pytest tests/test_data.py` passes including the
no-leakage assertion; `docs/data_report.md` exists.

### Day 4 — Fine-tuning + baselines
1. Kick off the large-baseline arm early if it's slow — run it in the
   background rather than waiting until the last day.
2. Start the QLoRA fine-tune (Phase 5); log peak VRAM every run.
3. Scaffold the classic/non-LLM baseline (Phase 6 arm 4) — do not skip it.

### Day 5 — First real numbers, or a clean handoff checklist
1. Run zero-shot eval on the held-out set → Phase 2 acceptance test.
2. Update `docs/DECISIONS.md` with everything decided across the week and
   why.
3. If blocked on anything external (data access, compute), write a
   "resume here" checklist in `docs/DECISIONS.md` with the exact commands
   to run the moment the blocker clears — this is the honest state to
   report, not a failure to hit a timeline.
