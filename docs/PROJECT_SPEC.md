# PROJECT SPEC — On-Prem Fine-Tuning: Small QLoRA Model vs. Larger Baseline on Company Data

**Read this entire file before writing any code.**
**Do not begin Phase 1 until the `OPEN_DECISIONS` block at the bottom is filled in by the user.**

This spec is cloned from `clinical-coding-eval`'s `PROJECT_SPEC.md`, which
proved a clean phase-gated pattern for exactly this shape of project: small
on-prem model vs. large baseline, fine-tuned on data that cannot leave the
building. The task-specific sections below are placeholders — **do not fill
them in by guessing.** Fill in `OPEN_DECISIONS` first; the rest of this
document should be rewritten around those answers before Phase 1 starts.

---

## 0. Thesis

> Can a small parameter-count model (Qwen2.5-3B-Instruct), QLoRA fine-tuned
> on a single 12GB consumer GPU, match or beat a larger off-the-shelf model
> at **generating a realistic clinical doctor-patient dialogue from a
> clinical note** (NoteChat) — under the hard constraint that all inference
> on the underlying clinical note data runs locally, never sent to a
> third-party API?

**Task pivot (locked in 2026-08-24):** this spec was originally written
around CUAD contract-clause extraction (see `docs/DECISIONS.md`'s Day 0
entries for that reasoning — kept for the record, not deleted). The data
pipeline actually built and the task actually trained (`configs/data.yaml`,
`src/data/build_dataset.py`, `notebooks/finetune.ipynb`) is NoteChat
instead. `docs/DECISIONS.md`'s "Task pivot" entry records why. This
document has been rewritten to match the NoteChat task, which is now the
authoritative one — do not revert `src/train/train.py` or this file back to
CUAD without a new decision entry.

The headline deliverable is **not** the fine-tuned model. It is the
**evaluation harness**. The model is evidence that the harness works.

---

## 1. HARD CONSTRAINTS — violating any of these invalidates the project

### 1.1 Data governance (non-negotiable, defaults to strictest until `OPEN_DECISIONS` says otherwise)

- **NEVER** write code that transmits company data, or any derivative
  containing it, to a third-party API (OpenAI, Anthropic, Groq, Together,
  Cohere, HuggingFace Inference API, or any hosted endpoint) unless
  `OPEN_DECISIONS` #3 explicitly permits it for a named subset of the data.
- **NEVER** paste real company data into a commit message, issue, log file
  that gets committed, README, or example output.
- **NEVER** commit any file under `data/` to git. Enforce with `.gitignore`
  AND the pre-commit hook (`scripts/check_data_leak.py`).
- All inference on sensitive data runs **locally** unless stated otherwise.
- If a task appears to require sending company data to an external service,
  **stop and ask the user**. Do not improvise.

### 1.2 Scientific integrity

- **Never write a number into a README, model card, or results table that
  was not produced by a script in this repo.** No placeholder metrics. No
  "expected ~0.75". If a result does not exist yet, write `TBD` or leave the
  cell empty.
- Every metric must be reproducible by a single documented command.
- If ground truth has a natural grouping key (customer, account, document
  source, time period) that could leak across train/val/test, split on that
  key, not on the raw row. Identify the correct key in `OPEN_DECISIONS` #5.
- Report confidence intervals, not bare point estimates.
- If a baseline beats the fine-tuned model, report it prominently. That is a
  finding, not a failure.

### 1.3 Style

- Write for a reader who will interrogate every choice in a job interview.
  Every non-obvious decision gets a one-line comment explaining *why*, not
  *what*.
- Prefer boring, legible code over clever code.
- No unnecessary abstraction layers. This is a research repo, not a
  platform.

---

## 2. Environment

```
GPU:        see OPEN_DECISIONS #4
System RAM: see OPEN_DECISIONS #4
OS:         see OPEN_DECISIONS #4
Python:     3.11
Package mgr: uv
```

Core stack: `torch`, `transformers`, `peft`, `bitsandbytes`, `trl`,
`unsloth`, `datasets`, `accelerate`, `polars` (or pandas), `duckdb`,
`scikit-learn`, `numpy`, `scipy`, `matplotlib`, `pytest`, plain YAML configs.

For the large-model baseline: `llama-cpp-python` with GPU offload, or `vllm`
if it fits.

**Pin every version.** bitsandbytes + unsloth + torch version drift is the
single most common cause of a broken week. Use `uv.lock` as the actual pin
(see `clinical-coding-eval`'s `docs/DECISIONS.md` for why lower-bound
constraints + lockfile beat hand-typed `==` pins).

---

## 3. Repo structure

```
notechat-qlora-eval/
├── README.md                  # results table (TBD until produced), repro commands
├── pyproject.toml
├── .gitignore                 # data/, *.parquet, *.csv, models/, outputs/
├── .pre-commit-config.yaml    # data-leak grep hook
├── configs/
│   ├── data.yaml
│   ├── train.yaml
│   └── eval.yaml
├── src/
│   ├── data/
│   │   └── build_dataset.py       # raw company data → task-ready parquet
│   ├── train/
│   │   └── qlora_sft.py
│   ├── inference/
│   │   ├── local_hf.py            # transformers/vLLM runner
│   │   ├── local_gguf.py          # llama.cpp runner for the large baseline
│   │   └── constrained.py         # schema-constrained decoding, if output is structured
│   ├── baselines/
│   │   └── classic_baseline.py    # non-LLM baseline (encoder / rules / regex — task dependent)
│   └── eval/
│       ├── metrics.py             # task-appropriate metric(s), bootstrap CIs
│       ├── calibration.py         # ECE, reliability curves (if applicable)
│       ├── selective.py           # risk-coverage, AUARC (if applicable)
│       ├── schema.py              # output-format validity rate (if structured output)
│       ├── grounding.py           # evidence/citation verification (if applicable)
│       ├── contamination.py       # memorization probe
│       └── run_eval.py            # single entrypoint
├── annotation/
│   ├── sample_for_annotation.py
│   ├── guidelines.md              # written BEFORE annotating
│   └── agreement.py               # Cohen's kappa
├── tests/
└── docs/
    ├── DECISIONS.md               # running log of every design choice + rationale
    └── MODEL_CARD.md
```

`docs/DECISIONS.md` is mandatory and updated at every phase. It is the
interview script. Not every module above will apply to every task — delete
what doesn't apply once `OPEN_DECISIONS` is filled in, and say why in
`docs/DECISIONS.md`.

---

## 4. Task definition

**Input:** the full text of one clinical note (NoteChat corpus, sourced
from PMC-Patients case reports).
**Output:** a realistic doctor-patient dialogue, formatted as alternating
`Doctor:`/`Patient:` turns, consistent with that note.

This is free-form generation, not extraction — there is no fixed label
space and no verbatim-evidence-span requirement, so `src/eval/grounding.py`
and `src/eval/schema.py` do not apply to this task (`configs/eval.yaml`'s
`schema.enabled: false` reflects this; delete those two modules rather than
stub them further if the task stays this shape). The eval harness instead
needs reference-based generation metrics (ROUGE/BERTScore against the
`conversation` column) plus whatever qualitative/structural checks matter
for this task (e.g. does the output actually alternate `Doctor:`/`Patient:`
turns, does it stay on-topic with the note) — see §4.2's caveat on what
those reference-based metrics can and cannot claim.

### 4.1 Label / output space

None — free-form dialogue generation, not classification/extraction. There
is nothing to report a per-category frequency table for; §4.1 exists in
this document only to note explicitly that it does not apply, per
`configs/data.yaml`'s `label_space.strategy: none`.

### 4.2 Known label-noise / ground-truth caveats

The `conversation` target is **not** a human-authored or human-verified
reference — it is itself the output of NoteChat's multi-agent LLM pipeline,
synthesized backwards from the (real) clinical note. See
`docs/data_report.md`'s "Ground-truth caveat" and `docs/about_dataset.md`
for the plain-language version.

**Consequence for evaluation:** any reference-based metric (ROUGE, BLEU,
BERTScore) computed against this column measures *similarity to one
synthetic exemplar*, not correctness against a real transcript or an
expert-verified answer, unlike CUAD's expert-annotated spans. Report this
caveat alongside any such metric in `README.md`/`docs/MODEL_CARD.md`, not
just in this spec. Phase 4 (gold annotation pass) is **not** a substitute
for this caveat — a human annotator grading the model's output against a
synthetic reference does not make the reference itself more authoritative.

---

## 5. Build phases

Each phase has an acceptance test. Do not advance until it passes. Work
phase by phase — do not scaffold all phases at once.

### Phase 1 — Data pipeline

1. Load raw company data from the source(s) named in `OPEN_DECISIONS` #2.
2. Build the task-ready dataset: join/filter to the in-scope population,
   derive the label/output space (§4.1).
3. Splits: split on the correct grouping key (`OPEN_DECISIONS` #5) to avoid
   leakage. Assert zero overlap across splits in a test.
4. Deduplicate near-identical records if applicable. Report how many were
   removed.

**Acceptance:** `pytest tests/test_data.py` passes, including a
no-leakage assertion. A `docs/data_report.md` exists with join rates,
coverage, distribution stats, and dedup counts.

### Phase 2 — Eval harness (BEFORE any training)

Build and validate the harness against zero-shot baselines only. Metrics
before models — this prevents unconsciously tuning the metric to flatter
the model. Implement whichever of these apply to the task:

| Metric | Detail |
|---|---|
| Task metric(s) (F1 / accuracy / ROUGE / etc.) | Per-class and aggregate, as applicable |
| Bootstrap CIs | 1000 resamples at the record level, 95% percentile CI |
| Paired bootstrap | For model-A-vs-model-B deltas. Report the delta CI. |
| Schema validity | % of raw generations that parse as valid structured output, if applicable |
| Grounding | % of evidence/citations verifiable against the input, if applicable |
| Cost / latency | tok/s, peak VRAM, wall-clock, watt-hours, derived cost per 1000 records |

**Acceptance:** `run_eval.py` produces a complete results JSON for a
zero-shot small-model run on a held-out sample. Every metric populated.
Unit tests for `metrics.py` against hand-computed toy examples.

### Phase 3 — Contamination probe

If the base model may have seen this data in pretraining (e.g. it is public
or partially public company content), test for memorization before claiming
a fine-tune taught the model anything. If the data is entirely private and
never public, this phase may be skipped — say so explicitly in
`docs/DECISIONS.md`, do not silently drop it.

**Acceptance:** `docs/contamination_report.md` with method and results, or
a `DECISIONS.md` entry explaining why it was skipped.

### Phase 4 — Gold annotation set

If ground truth is noisy (§4.2), sample a stratified set and get an
independent human label pass. Write `annotation/guidelines.md` **before**
annotating. Compute Cohen's kappa on an overlap sample.

**Acceptance:** gold set parquet (gitignored), kappa reported in
`docs/annotation_report.md`. Skip with a documented reason if ground truth
is already high-trust.

### Phase 5 — QLoRA fine-tuning

Model: [see `OPEN_DECISIONS` #1/#4 — confirm availability and exact
checkpoint name at build time, do not assume].

Starting config — treat as a starting point, not gospel (values below are
`clinical-coding-eval`'s starting point for a ~4B model on a 12GB GPU;
rescale to the actual model size and GPU from `OPEN_DECISIONS` #4):

```yaml
quantization: nf4, double_quant=true, compute_dtype=bfloat16
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
max_seq_len: 4096
per_device_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
optim: paged_adamw_8bit
lr: 2e-4
scheduler: cosine
warmup_ratio: 0.03
epochs: 2
packing: true
```

- Use Unsloth on a VRAM-constrained card — roughly 2x throughput and
  materially lower VRAM.
- Log peak VRAM every run. If OOM: reduce `max_seq_len` before touching
  rank or batch size.
- Ablations to run (each is an interview talking point): LoRA rank
  {8, 16, 32}; input formatting variants; 1 vs. 2 epochs; prompt format
  variants.

**Acceptance:** training run reproducible from a config file; adapter
saved; eval harness produces a full result set on the test split.

### Phase 6 — Baselines

At minimum:

1. **Zero-shot small model** — floor.
2. **Zero-shot large open-weight baseline** — slow but runs once over the
   test set.
3. **QLoRA'd small model** — the contribution.
4. **Classic/non-LLM baseline** (encoder classifier, rules engine, regex,
   whatever is the field's standard tool for this exact task). Do not skip
   this arm — "the right tool for this task is not an LLM" is a legitimate
   and valuable finding if it turns out to be true.

### Phase 7 — Write-up

- `README.md`: thesis, constraints, results table with CIs, repro commands,
  honest limitations section.
- `docs/MODEL_CARD.md`: intended use, out-of-scope use, training data
  description (no raw sensitive content), evaluation results,
  contamination findings, label-noise caveat.

---

## 6. Instructions for you (Claude Code)

- **Ask, don't assume.** If a checkpoint name, API, library version, data
  schema, or sensitivity classification is uncertain, say so and ask. Do
  not write plausible-looking code against a guessed schema.
- **Verify checkpoint names and library APIs at build time.** This spec was
  written from knowledge that may be stale.
- Work phase by phase. Do not scaffold all phases at once.
- After each phase, append to `docs/DECISIONS.md`: what was decided, what
  alternatives were considered, why this one.
- Write the test before or alongside the code for anything in `src/data/`
  and `src/eval/`. Silent data bugs are the primary risk in this project.
- Prefer failing loudly over defaulting silently.

---

## 7. OPEN_DECISIONS — filled in Day 0, see `docs/DECISIONS.md` for rationale

**Superseded by the task pivot to NoteChat (2026-08-24) — kept below for
the historical record per `docs/DECISIONS.md`'s Day 0 entries. The current
answers are in the follow-up block immediately after.**

```
1. [SUPERSEDED] Task type & output shape:
   Extraction, multi-label. Input: one commercial contract's text.
   Output: a list of {clause_type, evidence_span} pairs, one per clause
   type present, where evidence_span is a verbatim substring of the
   input contract. 41 fixed clause categories (CUAD's taxonomy — e.g.
   Governing Law, Termination for Convenience, Non-Compete, IP
   Ownership Assignment, Most Favored Nation, ...). Same shape as
   clinical-coding-eval's label+evidence pattern.

2. [SUPERSEDED] Data description:
   CUAD (Contract Understanding Atticus Dataset) — 510 real commercial
   legal contracts (license, distributor, IP assignment, services,
   etc.), expert-annotated (law-student annotators supervised by
   experienced lawyers, via the Atticus Project) for 41 clause
   categories, ~13,000 label instances. Public dataset, distributed as
   SQuAD-style JSON (question = clause type, context = contract text,
   answer = verbatim span). Source: HuggingFace `theatticusproject/cuad-qa`,
   or the original release at https://www.atticusprojectai.org/cuad.
   ~500MB including source PDFs/txt.

3. [SUPERSEDED] Sensitivity / compliance constraints:
   None — CUAD is public data, not real confidential company data. It
   stands in for "our own company data" so the governance-first
   pipeline and eval harness can be built and demonstrated end-to-end.
   Because there is no real confidentiality requirement, §1.1's
   "never leaves the machine" rule is kept as the *default working
   posture* for realism (the harness is the deliverable, and it should
   be built as if the constraint were real) but is not a legal
   requirement for this dataset specifically — noted here so it is
   never misrepresented as an actual compliance finding. Precisely
   because the data is public, Phase 3 (contamination probe) is NOT
   skipped — it is a real, meaningful check, since a foundation model
   may have seen CUAD during pretraining.

4. Compute:
   Local machine (this repo): no GPU, 31GB RAM, Linux — used for the
   data pipeline (Phase 1), eval harness (Phase 2), contamination probe
   (Phase 3), and analysis. QLoRA training (Phase 5) runs on a separate
   GPU machine the user owns: 12GB VRAM, 32GB system RAM. The starting
   config in §5 (r=16, seq_len=4096, batch=1, grad_accum=16) targets
   roughly this class of card (originally sized for clinical-coding-eval's
   ~4B model on 12GB) — re-check actual VRAM headroom once the model size
   for Phase 5 is picked, reduce `max_seq_len` first if OOM. The trained
   adapter is synced back to this machine for eval.

5. [SUPERSEDED] Leakage-safe split key:
   contract_id (i.e. source document). A single contract carries many
   clause-type annotations; splitting at the clause/label-instance
   level would leak the same contract's language across train/val/test.
   Split 80/10/10 by contract_id, seed 42 (configs/data.yaml).

6. [SUPERSEDED] Ground truth / labels:
   Already exists and is high-trust — CUAD annotations were produced by
   a structured legal-expert annotation process (Atticus Project),
   not scraped or heuristic labels. Treated as clean ground truth, not
   a noisy silver label. Phase 4 (gold annotation pass) is skipped;
   see `docs/DECISIONS.md` for the documented reasoning this section
   requires.

7. Baseline model to compare against:
   A larger open-weight instruction-tuned model run locally via
   llama.cpp/GGUF with GPU offload on the training machine (e.g. a
   Llama-3.1 or Qwen2.5 instruct checkpoint in the 32B-70B range).
   Exact checkpoint name TBD — verify availability and license at
   build time per §6, do not assume. Still applies unchanged under the
   NoteChat pivot below.
```

### 7a. OPEN_DECISIONS — current answers (NoteChat, locked in 2026-08-24)

```
1. Task type & output shape:
   Free-form generation, not extraction. Input: one clinical note's
   full text. Output: a doctor-patient dialogue, formatted as
   alternating `Doctor:`/`Patient:` turns, consistent with that note.
   No fixed output taxonomy (contrast CUAD's 41 clause categories) —
   see §4.1.

2. Data description:
   NoteChat — clinical notes are PMC-Patients case reports (public,
   sourced from PubMed Central); the paired `conversation` is
   NoteChat's own multi-agent LLM pipeline output, synthesizing a
   plausible doctor-patient dialogue backwards from each note (not a
   real transcript — see §4.2's ground-truth caveat). Using the
   `our_revised_v2.csv` release variant (10,000 notes) — see
   `docs/DECISIONS.md` for why this variant over NoteChat's other
   released files (`gpt-3.5.csv`, `gpt-4.csv`, `gpt-3.5-160k.csv`).
   Source file lives at `data/raw/notechat/our_revised_v2.csv`
   (gitignored, per §1.1).

3. Sensitivity / compliance constraints:
   None — NoteChat's notes and dialogues are both public research
   data, not real confidential company or patient data (this is
   synthetic/case-report data, not real PHI). As with the original
   CUAD framing, it stands in for "our own company data" so the
   governance-first pipeline is built and demonstrated as if the
   constraint were real, even though it is not a legal requirement for
   this dataset specifically. Because the data is public, Phase 3
   (contamination probe) is not skipped for the same reason item #3
   above gave for CUAD.

4. Compute: unchanged — see item 4 above (this section was never
   CUAD-specific).

5. Leakage-safe split key:
   note_id (one row = one unique clinical note, verified at build
   time — no note appears more than once). Split 80/10/10 by note_id,
   seed 42 (configs/data.yaml). Simpler than CUAD's case: NoteChat has
   no multi-row-per-document structure to worry about, since each row
   already is one document.

6. Ground truth / labels:
   NOT high-trust in the way CUAD's were. The `conversation` target is
   itself LLM-generated (NoteChat's pipeline), not human-authored or
   human-verified — see §4.2. This is the opposite situation from
   CUAD's expert-annotated spans, and every downstream reference-based
   metric must carry this caveat. Phase 4 (gold annotation pass) is
   still skipped, but for a different reason than CUAD's: there is no
   practical way to get an "expert-authored" doctor-patient dialogue
   at this dataset's scale, so a small human gold set would only ever
   validate *a* plausible dialogue, not correct the target column's
   caveat above. If Phase 4 is revisited, frame it as human plausibility
   rating, not as gold-label correction.
```

---

## 8. Explicit non-goals

- No web UI. No FastAPI service. No Docker orchestration. No agent
  framework — unless `OPEN_DECISIONS` explicitly calls for one.
- This project is judged on measurement rigor, not on deployment surface
  area.
- Do not add RAG unless the task genuinely requires retrieval and
  `OPEN_DECISIONS` says so.
