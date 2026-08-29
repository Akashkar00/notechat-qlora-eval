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

**Phase 2 acceptance results (2026-08-25):** both arms completed on the
same 200-record sample of `data/processed/test.parquet` (seed 42).
Zero-shot: ROUGE-1 0.291, ROUGE-2 0.079, ROUGE-L 0.149, BERTScore F1 0.852,
turn-format-valid 74.5%, 25.0 tok/s. Fine-tuned: ROUGE-1 0.631, ROUGE-2
0.349, ROUGE-L 0.405, BERTScore F1 0.909, turn-format-valid 100%, 17.2
tok/s (slower — consistently produces longer, fuller dialogues rather than
stopping early). Paired bootstrap deltas (finetuned − zero-shot, all 95%
CIs exclude zero): ROUGE-1 +0.341 [+0.321, +0.359], ROUGE-2 +0.269 [+0.255,
+0.285], ROUGE-L +0.257 [+0.241, +0.272], BERTScore F1 +0.057 [+0.054,
+0.060]. Full per-arm records in `artifacts/eval/{zero-shot,finetuned}/
results.json`; paired comparison in
`artifacts/eval/comparison_zero_shot_vs_finetuned.json`. Caveat carried
forward from §4.2: this is similarity to NoteChat's own LLM-generated
reference, not ground-truth correctness — a real, large, statistically
solid effect on that similarity measure, not a claim of medical accuracy.

**Still open after this run:** this only shows fine-tuning helped the small
model beat its own zero-shot baseline — it says nothing yet about whether
the fine-tuned 3B model is competitive with a larger model (Phase 6 arm 2,
not built) or a classic/non-LLM approach (arm 4, not built). Those are the
comparisons the project's actual thesis depends on.

---

## Phase 6 arm 2 — large baseline model, serving backend pivot (2026-08-25)

**Decision:** picked `unsloth/Qwen2.5-14B-Instruct-bnb-4bit` as the "larger
open-weight baseline" (`OPEN_DECISIONS` #7 was left as "32B-70B range,
verify at build time" — 14B is the size that's actually feasible on this
12GB-VRAM/32GB-RAM machine; confirmed the checkpoint exists on HuggingFace
before using it, per §6). Served it through the **same bnb-4bit
transformers/unsloth stack already used for the fine-tune**
(`src/inference/local_hf.py`'s existing `load_model`/`generate`, unchanged —
just a bigger `--model-name`), not llama.cpp/GGUF as `PROJECT_SPEC.md` §5
originally suggested.

**Why the pivot:** `llama-cpp-python`'s only CUDA-enabled wheels for this
platform (cp311/win_amd64) are hosted as GitHub release assets
(`abetlen.github.io/llama-cpp-python/whl/cu124`, resolving to
`release-assets.githubusercontent.com`) — that specific host is unreachable
from this machine's network (confirmed: `github.com`, `raw.githubusercontent.com`,
and `huggingface.co` all resolve fine; `release-assets.githubusercontent.com`
times out on TLS connect, both via `uv` and raw `curl`). PyPI ships no
prebuilt wheel at all for this package (source-only), and no C++
toolchain/CMake is installed to build it locally. Rather than spend
significant time installing Visual Studio Build Tools just to compile a
CPU-only or CUDA fallback, reused the transformers/bitsandbytes/unsloth
stack that's already proven working on this exact machine — same
quantization approach (bnb NF4 4-bit) as the fine-tune, so this isn't a
lower-effort/lower-quality substitute, just a different (already-verified)
serving backend for the same "run it locally, 4-bit-quantized" idea the
spec wanted.

**Feasibility check (2-record smoke test before the real run):** peak VRAM
10.42GB/12GB — tight but fits; `max_seq_len=2048` override used (via new
`run_eval.py --max-seq-len` flag) instead of `configs/train.yaml`'s 4096,
since that value was sized for the 3B fine-tune's training context, not a
14B zero-shot eval, and clinical notes are short enough (max 2,330 chars,
`docs/data_report.md`) that 2048 tokens is ample headroom. Throughput:
5.4 tok/s, ~53s/record — real full-200-record estimate ~2.9 hours (much
slower than QLoRA's 3B arms, as expected for ~4.7x the parameters).
Zero-shot turn-format-valid rate on the smoke sample was already 100%
(vs. the 3B zero-shot's 74.5%) — a 14B model following the format
instruction reliably even without fine-tuning is itself a plausible
finding, to be confirmed on the full run.

**Consequence for `OPEN_DECISIONS` #7 / `ARCHITECTURE.md`:** both still
describe "llama.cpp/GGUF with GPU offload" — noted here rather than fixed
silently; needs updating alongside the broader `ARCHITECTURE.md`/`README.md`
rewrite already tracked in `docs/STATUS_AND_ROADMAP.md`.

**Full 200-record run, results (2026-08-25):** ROUGE-1 0.445, ROUGE-2
0.117, ROUGE-L 0.198, BERTScore F1 0.865, turn-format-valid 100%, 12.3
tok/s, peak VRAM 10.44GB. Saved to
`artifacts/eval/large-baseline-zero-shot/results.json`.

**This is the project's headline result.** Paired bootstrap comparison
across all three arms (`artifacts/eval/comparison_all_arms.json`), same 200
test records:

| Comparison | ROUGE-1 Δ | ROUGE-2 Δ | ROUGE-L Δ | BERTScore Δ |
|---|---|---|---|---|
| fine-tuned 3B − zero-shot 14B | +0.186 [+0.174, +0.198] | +0.232 [+0.216, +0.248] | +0.208 [+0.193, +0.223] | +0.043 [+0.040, +0.047] |
| fine-tuned 3B − zero-shot 3B | +0.341 [+0.321, +0.359] | +0.269 [+0.255, +0.285] | +0.257 [+0.241, +0.272] | +0.057 [+0.054, +0.060] |
| zero-shot 14B − zero-shot 3B | +0.154 [+0.136, +0.173] | +0.038 [+0.030, +0.045] | +0.049 [+0.041, +0.058] | +0.014 [+0.012, +0.016] |

**The fine-tuned 3B model beats the zero-shot 14B model (~4.7x more
parameters) on every metric, every CI excluding zero.** Scaling 3B→14B
zero-shot bought +0.154 ROUGE-1; QLoRA fine-tuning the 3B model bought
+0.341 — more than double the effect of a ~5x parameter increase, on this
task and this similarity measure.

**How strongly this should be stated (revised 2026-08-27).** An earlier
version of this entry called it "the thesis in `PROJECT_SPEC.md` §0
confirmed." That overstates it, and the overstatement is worth correcting
in place rather than quietly leaving. What was actually measured is
*similarity to NoteChat's house style*, and arm 3 trained on 8,000 examples
of exactly that style while the 14B model has never seen it. Winning that
comparison is close to structurally guaranteed and is not the same claim as
"a 3B model is better at generating clinical dialogue." The honest
statement is narrower:

> QLoRA fine-tuning adapts a 3B model to a target corpus's format and style
> far more effectively than a ~5x parameter increase does, at a fraction of
> the serving cost.

That is still a genuinely useful, non-obvious, and correctly-evidenced
finding — it just isn't a claim about general capability. Arm 4 (below)
tests the same skepticism from the other direction.

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

---

## Rigor pass — greedy decoding, faithfulness, arm 4, Phase 3 (2026-08-27)

A deliberate pass over the project's own weak points rather than new
features. Each item below was a thing an informed reader could have
attacked, listed with what was done about it.

**1. Evaluation was stochastic, so no number was reproducible.**
`local_hf.generate` defaulted to `do_sample=True, temperature=0.7`. That
quietly broke two claims this repo makes. `PROJECT_SPEC.md` §1.2 requires
every metric to be reproducible by a single documented command — but a
rerun produced different numbers. And `metrics.bootstrap_ci` resamples
records as though each record's score were a fixed quantity, when it was
actually one draw from a distribution, so the published intervals
understated true uncertainty. Changed the default to greedy
(`do_sample=False`), added an explicit `--sample` opt-in, recorded the
decode config inside every `results.json`, and made `compare.py` print a
warning if it is ever handed a stochastically-decoded arm. **All arms were
re-run from scratch under greedy decoding**; the numbers in `README.md` are
from that rerun, and the earlier sampled numbers are superseded.

**2. The comparison numbers were the one thing nobody could regenerate.**
The paired deltas in `README.md` came from a throwaway script, violating
§1.2 in the middle of the results table. Added `src/eval/compare.py` (`python
-m src.eval.compare`), which discovers every `artifacts/eval/*/results.json`,
**asserts the arms were scored on identical records before comparing**
(silently intersecting mismatched runs would produce a confident-looking
delta over whatever happened to overlap), computes all pairwise deltas, and
emits both `comparison.json` and the `comparison.md` table.

**3. A known failure mode was documented but never measured.** The Phase 5
sanity check caught the fine-tuned model inventing an ungrounded aside, and
that finding sat as a footnote. Every metric in the harness compared
generations to the *reference dialogue*, which is itself LLM-generated
(§4.2) — so nothing could see fabrication at all. Added
`src/eval/faithfulness.py`, which scores against the **clinical note**, the
one real artifact in the dataset: `numeric_grounding_recall` (did the
dialogue carry over the note's numbers), `numeric_precision`, and
`fabricated_number_rate`. Numbers were chosen because clinical notes are
number-dense and a numeric claim is unambiguously checkable, unlike
paraphrase. Deliberately *not* an LLM-as-judge, which would reintroduce the
exact "grade one model with another model" problem §4.2 warns about. It is
a proxy and is documented as one.

**4. Arm 4 existed only as an excuse.** `src/baselines/` was an empty
`__init__.py` while the write-up claimed a four-arm comparison. Added
`TfidfRetrievalBaseline`: fit TF-IDF on the training notes, and for a test
note return the conversation attached to the most cosine-similar training
note. No model, no GPU, deterministic. Fitted on **train only** — retrieving
from a pool containing the query would make the arm trivially and
meaninglessly perfect.

**This arm produced the most interesting result in the project.** On a
5-record smoke test it scored ROUGE-1 ~0.46 — *higher than the zero-shot
14B model's 0.445* — while fabricating ~59% of the numbers it stated and
carrying over only ~14% of the note's own numbers. A method with no model
at all beats a 14B LLM on the headline metric, because the retrieved
dialogue is fluent, correctly formatted, on-topic, and about **the wrong
patient**. That is direct evidence that ROUGE/BERTScore on this task
substantially measure style-matching rather than quality, which is exactly
the skepticism §5 Phase 6 asks arm 4 to test, and it is the strongest
available justification for having built the faithfulness metric.

**5. Phase 3 was never started.** Added `src/eval/contamination.py`.
NoteChat is public and its notes come from PMC-Patients, so the base model
may have memorized the corpus — which would mean fine-tuning gains partly
reflect recall, not learning. Method: feed the **base** model the first half
of a held-out dialogue as raw text (no chat template — probing the LM's
memory, not its instruction-following), continue greedily, and measure
ROUGE-L against the true second half. Crucially it scores the *same*
generation against a **randomly chosen other** dialogue's second half as a
control: NoteChat dialogues are formulaic, so any competent model scores
well above zero without memorizing anything, and only the gap between true
and control is evidence. Reported as a paired bootstrap delta with a CI, so
"no evidence of contamination" is measured rather than asserted.

**6. `src/train/train.py` still described CUAD.** The longest-standing open
item in this log. Rewritten for NoteChat, and it now **imports
`SYSTEM_PROMPT` / `build_user_prompt` from `src/inference/local_hf.py`
instead of defining its own copies** — the duplicate definitions are what
let training and evaluation drift apart in the first place. Also added an
explicit `trainer.evaluate()` after `trainer.train()`, because the notebook
run of this same config left no epoch-2 `eval_loss` in `trainer_state.json`
(open item below), making the final validation loss unrecoverable without
retraining; an explicit call means that number always exists regardless of
callback scheduling. Training history is now persisted to
`train_history.json` next to the adapter, which survives `save_total_limit`
pruning the checkpoint dirs.

**7. "Why 200 of 1,000 test records?" had no answer.** Now justified
quantitatively in `configs/eval.yaml`: at n=200 the observed 95% CI
half-width on ROUGE-1 is ~±0.008 against effect sizes of +0.19 to +0.34 —
25-40x the interval width. Scoring all 1,000 would cost 5x the GPU time
(the 14B arm alone: ~1.6h → ~8h) to narrow intervals by only sqrt(5) ≈ 2.2x,
moving no conclusion.

**8. Windows encoding bug, found by running the new code.**
`compare.py`'s report write crashed with `UnicodeEncodeError` because
Windows defaults `write_text` to cp1252, which cannot encode the minus sign
the table uses — the whole comparison would have died at its last step
after a 3.5-hour run. All report/results writes now pass
`encoding="utf-8"` explicitly. Same class of latent Windows bug as the
hardcoded `/tmp` path fixed earlier.

**Test suite: 24 → 46 tests**, all passing. New coverage: numeric
faithfulness (including hand-computed partial-overlap and the vacuous
no-numbers cases), the retrieval baseline (topical correctness,
determinism, fit-before-predict failing loudly, and a test pinning the
*structural weakness* that it carries the wrong patient's numbers), and
`compare.py`'s mismatched-record guard.

**Still open after this pass:** the missing epoch-2 `eval_loss` in the
existing adapter's `trainer_state.json` is still unexplained — the rewritten
CLI trainer prevents a recurrence, but diagnosing the original would require
retraining, which is not worth 4+ GPU-hours to recover one number that the
held-out test evaluation supersedes anyway. `docs/ARCHITECTURE.md` still
carries pre-pivot CUAD diagrams. No LoRA rank / epoch-count ablations have
been run.

---

## Final greedy 4-arm results (2026-08-28)

The greedy rerun launched during the rigor pass was interrupted mid-run
(background process died after arms 1 and 3 completed, before arm 2). Arm 2
was relaunched standalone and completed; `compare.py` was then re-run over
all four fresh, mutually-comparable, greedy-decoded arms. These are the
numbers now in `README.md`, superseding every prior entry in this log that
cited sampled-decoding figures.

**Per-arm** (full table: `README.md` Results, raw: `artifacts/eval/*/results.json`):

| Arm | ROUGE-1 | BERTScore-F1 | Numeric grounding recall | Fabricated numbers |
|---|---|---|---|---|
| Zero-shot 3B | 0.289 | 0.853 | 0.069 | 0.026 |
| Zero-shot 14B | 0.442 | 0.866 | 0.058 | 0.020 |
| Fine-tuned 3B | 0.635 | 0.910 | **0.478** | 0.007 |
| TF-IDF retrieval | 0.461 | 0.863 | 0.145 | **0.699** |

**Confirms every conclusion drawn from the sampled/smoke-test numbers,
with one addition the smoke test couldn't show:**

1. Fine-tuned 3B beats zero-shot 14B on every reference-similarity metric,
   all CIs excluding zero (ROUGE-1 Δ +0.193 [+0.181, +0.206]) — matches the
   sampled run's +0.186 almost exactly, confirming decoding variance was
   never the source of this effect.
2. TF-IDF retrieval (200 records, not the earlier 5-record smoke test)
   significantly outscores zero-shot 14B on ROUGE-1: Δ +0.020
   [+0.010, +0.029] — small but the CI is entirely positive, so "matched" in
   the earlier write-up undersold it. Now stated as "outscored" in
   `README.md` and `CV.tex`. Fabrication rate on the full 200: **69.9%**
   [65.5%, 74.1%], not the smoke test's rough 59% — the real number is
   worse than the estimate that shipped in the CV first, corrected there
   too.
3. **New finding the smoke test was too small to surface:** fine-tuning's
   numeric-grounding-recall gain over zero-shot 14B is +0.420
   [+0.382, +0.459] — a ~7x relative increase (7%→48%), the single largest
   effect size in the whole comparison table. This is a genuine content-level
   improvement, not style-matching, since faking it would require actually
   using the query note's specifics. Correcting an earlier overcaution in
   this log: the Phase 6 arm 2 entry above said the fine-tuning win
   "substantially reflects learned house style... not a claim about general
   capability." That's still the right read for the ROUGE/BERTScore gains,
   but it doesn't apply to the grounding-recall result, which is a
   capability claim the data actually supports.
4. **What did not move:** fabrication rate itself, fine-tuned vs. zero-shot
   14B, Δ −0.013 [−0.035, +0.005] — CI includes zero. Both arms fabricate a
   small, statistically indistinguishable share of the numbers they choose
   to state. Fine-tuning made the model state far more of the note's real
   numbers; it did not measurably reduce its rate of inventing wrong ones
   among what it does state. Reported as two separate, differently-evidenced
   claims in `README.md` rather than collapsed into one "faithfulness
   improved" headline.

**Process note:** the interrupted background run is why arm ordering in
`artifacts/eval/` timestamps looks discontinuous (arms 1, 3, 4 dated
2026-08-27; arm 2 and the final `comparison.json`/`comparison.md` dated
2026-08-28). No consequence for validity — `compare.py`'s `assert_comparable`
check passed on all four, confirming identical held-out records regardless
of when each arm was produced.

---

## 2026-08-28 — Review pass: correctness fixes, tooling, doc drift

An external review of the finished repo. Findings acted on, in order of how
much they mattered.

### 1. The contamination control was not a derangement (real bug)

**Decision:** replaced `rng.shuffle` with `derange()`, implemented as
Sattolo's algorithm, in `src/eval/contamination.py`.

**Why:** the control scores each generated continuation against *another*
dialogue's second half, establishing the "generic NoteChat style" floor. A
plain shuffle permits fixed points, so at n=100 there was a ~63% chance
(1 − 1/e) that at least one record was scored against its **own** true
continuation while being labelled the control. That inflates the control
floor and biases the measured delta toward "no contamination" — i.e. the bug
pushed the probe toward the reassuring answer, which is the worst direction
for it to be wrong in.

**Alternatives considered:** shuffle-then-rotate-by-one. Tried first and
**rejected because it is wrong**: rotation removes fixed points with respect
to the *shuffled* order, not with respect to the original index, which is the
one that matters. The unit test caught it immediately, which is the argument
for having written the test before trusting the reasoning. Sattolo's
algorithm draws uniformly from the cyclic permutations, every one of which is
a derangement, so the guarantee is structural rather than rejection-sampled.

### 2. The pre-commit hook blocked `src/data/` (real bug, never noticed)

**Decision:** anchored `check_data_leak.py`'s path rules to the repo root
(`BLOCKED_TOP_LEVEL_DIRS`) instead of testing `part in path.parts`.

**Why:** this is the **identical bug `.gitignore` already carried and had
fixed** — an unanchored `data` matches a directory at any depth, so the hook
flagged `src/data/build_dataset.py`, the entire Phase 1 pipeline. It went
unnoticed only because `pre-commit install` was evidently never run; the hook
would have rejected those commits the moment it was. Same root cause, two
files, found by running the hook against every tracked file rather than by
reading it.

### 3. `SUSPICIOUS_PATTERNS` was still an empty template placeholder

**Decision:** populated it with the raw NoteChat CSV header and PHI-shaped
identifiers (SSN, MRN, NHS number, labelled DOB, email). Added
`ALLOWLISTED_PATHS` for `artifacts/eval/`, and 10 tests.

**Why:** an empty pattern list in a security-flavoured hook is worse than no
hook — it reads as a control that exists. The patterns screen for what would
matter if this harness were pointed at a real clinical corpus, which is the
stated design constraint (§1.1), rather than at NoteChat specifically.

**Deliberately not matched:** `Doctor:`/`Patient:` turn markers. They appear
legitimately 25 times in `tests/test_data.py` alone; a hook that fires on
ordinary commits gets bypassed, at which point it protects nothing. Verified
by running the hook over every tracked file (clean) and over synthetic leak
cases (all blocked), including benign clinical numbers like "64 years old,
BP 135/85, seen on 2015-03-12" which must *not* trip it.

### 4. Ruff's import sorter would have broken training (caught in review)

**Decision:** adopted ruff (config in `pyproject.toml`, hooks in
`.pre-commit-config.yaml`) — but pinned the unsloth import order in
`src/train/train.py` behind `# noqa: I001`.

**Why:** `ruff --fix` alphabetised that block, putting `trl` above `unsloth`
and **silently disabling unsloth's patching** — the exact failure the comment
directly above those imports warns about. Worth recording as the general
lesson: an autofixer optimises for the rule it knows, not for the invariant
the comment states, and a lint tool introduced late will happily "fix" load-
bearing code. Every other ruff finding was accepted; the `zip(..., strict=)`
ones are genuine, since silent length-truncation is precisely the
index-alignment failure `compare.py` exists to prevent.

### 5. `tokens_per_second` was measured from re-tokenized text

**Decision:** `local_hf.generate(..., return_n_tokens=True)` now returns the
length of the generated ids; `run_eval.py` uses it instead of re-tokenizing
the decoded string.

**Why:** decoding drops special tokens and the re-encode is not guaranteed to
round-trip to the same segmentation, so the old count measured something
adjacent to, but not equal to, what the model emitted.

**Consequence, stated plainly:** the committed `tokens_per_second` figures
(16.0 and 11.5 tok/s in `README.md`) were produced by the old method and were
**not** regenerated — that would cost ~3.5 GPU-hours to move a throughput
figure slightly. They are flagged in `STATUS_AND_ROADMAP.md` as predating the
fix. No quality metric, delta, or CI is affected: the change touches only the
token count, and every conclusion in this repo rests on the others.

### 6. Doc drift — the README contradicted itself

**Decision:** rewrote the README status table and `STATUS_AND_ROADMAP.md`;
added `LICENSE`; corrected the claim that `docs/ARCHITECTURE.md` is
pre-pivot.

**Why:** the status table said arm 4 and the contamination probe were "not
built" while the Results section below it reported arm 4's numbers in full
and linked `docs/contamination_report.md`. A reader hits that contradiction
in the first thirty seconds and reasonably starts discounting everything
after it — the cheapest possible fix for the most expensive kind of doubt.
`ARCHITECTURE.md` had in fact been rewritten for NoteChat on 2026-08-27; the
README and roadmap were both still describing it as CUAD-era, so the
correction went the other way for once.

**Also:** Phase 3 had been *written but never run*, leaving a documented file
path that did not exist. Running it is what closed the phase.

### 7. Phase 3 result — contamination is detectable but small

**Finding:** on 100 held-out dialogues, the base model's continuations land
+0.0206 ROUGE-L closer to the true second half than to a deranged control,
95% CI [+0.0140, +0.0267]. The interval excludes zero.

**Decision:** report this as **"detectable but small"**, not as either
"evidence of memorization" or "no contamination", and rewrite
`write_report`'s verdict logic to be magnitude-aware rather than binary.

**Why:** the original verdict was `excludes_zero and delta > 0` → "Evidence
of memorization." That is technically true and practically misleading. With
n=100 the probe has the power to resolve an effect far smaller than one that
could undermine any conclusion here, so "the CI excludes zero" and "this
changes the result" are different claims and a report that merges them is
reporting statistical significance as if it were importance. The effect is
+0.021 ROUGE-L against a fine-tuning effect of +0.215 (arm 3 over arm 2) and
+0.266 over the very base model probed — an order of magnitude larger.

`MATERIAL_DELTA = 0.05` is now a named module constant, ~a quarter of the
fine-tuning effect, published as a number precisely because it is a judgement
call: a reader who disagrees can recompute from the table in the report
rather than having to reverse-engineer a threshold buried in prose.

**What this costs the project, honestly:** it can no longer say the corpus is
clean. The defensible sentence is "a small, real contamination signal exists
and is far too small to account for the result." That is weaker than the
result the probe was hoped to produce, and it is what the data supports.

**Worth noting the derangement fix pushed the result this way, not away from
it.** A shuffle with fixed points inflates the control floor, which shrinks
the measured delta. Had the original `rng.shuffle` been left in place, this
signal would have been biased toward disappearing — the bug's failure mode
was to hand back the reassuring answer.

---

## Voice input added — `scripts/voice_to_note.py` (2026-08-29)

**Decision:** added a voice-to-text front end for the existing
note-to-dialogue pipeline. `scripts/voice_to_note.py` transcribes an audio
file (`--file`) or a live microphone recording (`--mic`, stopped by pressing
Enter) with **faster-whisper**, running entirely on-device, then hands the
transcript to the same `load_model`/`generate` call `try_model.py` already
uses and prints through the same `report()` function — so a voice note is
scored by the identical numeric-grounding/turn-format checks as a typed one,
not a parallel, potentially-drifting code path. `pyproject.toml` gained a
`voice` extra (`faster-whisper`, `sounddevice`), isolated from the base
group the same way `gpu` is — the text-only pipeline
(`build_dataset`/`run_eval`/`try_model`) never needs an audio stack, and
`faster-whisper` alone pulls in `ctranslate2`, `onnxruntime`, and `av`.

**Why this doesn't touch `PROJECT_SPEC.md` §8's non-goals:** those bar a web
UI, a FastAPI service, Docker orchestration, and agent frameworks — none of
which this is. It's a third local CLI entrypoint alongside `try_model.py`
and `run_eval.py`, reusing their model-loading and scoring code rather than
adding a new surface to maintain.

**Local-only, deliberately:** transcription runs on-device via
`faster-whisper`; no audio or transcript is ever sent to an external API,
matching `PROJECT_SPEC.md` §1.1's data-locality rule. `--whisper-device
cuda` is supported for speed but is not required — CPU + `int8` is the
default so the script works without a GPU.

**Alternatives considered:** the original `openai-whisper` package —
rejected in favor of faster-whisper's CTranslate2 backend, which is faster
and lighter on this same 12GB-VRAM machine already carrying the QLoRA
stack; a hosted transcription API — rejected outright, since it would send
audio (potentially real patient speech, in a non-demo deployment) to a third
party, the exact thing §1.1 exists to prevent.

**Not yet done:** no unit tests were added — `record_from_mic`/`transcribe`
depend on a physical microphone/audio backend and aren't meaningfully
testable without one (unlike the deterministic model/metric code the test
suite covers); `uv.lock` was regenerated via `uv sync --extra gpu --extra
voice` and both the `gpu` and `voice` extras' packages were confirmed to
coexist correctly. No hyperparameter or size claim for the STT model itself
has been measured — `--whisper-model small` is Whisper's own recommended
default for the accuracy/speed tradeoff, not a value this repo benchmarked.
