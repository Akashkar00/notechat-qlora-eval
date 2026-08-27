# Architecture — NoteChat clinical dialogue generation

System architecture for the project defined in `PROJECT_SPEC.md`. Rewritten
2026-08-27 for the NoteChat task; earlier revisions of this file described
CUAD contract-clause extraction, which this project pivoted away from on
2026-08-24 (see `DECISIONS.md`, "Task pivot").

The invariant that survived the pivot: **everything runs on the local
machine.** No data, and no derivative of it, is sent to a third-party API
(`PROJECT_SPEC.md` §1.1).

---

## 1. Big picture — how data becomes a result

```mermaid
flowchart TD
    RAW["NoteChat our_revised_v2.csv<br/>10,000 clinical notes<br/>(gitignored)"] --> BUILD["Phase 1<br/>src/data/build_dataset.py"]
    BUILD --> SPLIT[("train 8,000 / val 1,000 / test 1,000<br/>split by note_id, seed 42")]
    BUILD --> DREPORT["docs/data_report.md"]

    SPLIT --> TRAIN["Phase 5<br/>QLoRA fine-tune"]
    TRAIN --> ADAPTER[("LoRA adapter")]

    SPLIT --> HARNESS["Phase 2<br/>Eval harness"]
    ADAPTER --> HARNESS
    HARNESS --> RESULTS["Phase 6<br/>4 arms compared"]
    RESULTS --> WRITEUP["Phase 7<br/>README, MODEL_CARD"]

    SPLIT --> CONTAM["Phase 3<br/>contamination probe"]
    CONTAM --> WRITEUP

    style RAW fill:#f8d7da,stroke:#b4545a,stroke-width:2px,color:#1a1a1a
    style WRITEUP fill:#d3edda,stroke:#3f8f57,stroke-width:2px,color:#1a1a1a
```

```mermaid
flowchart LR
    subgraph LOCAL["This machine only — no network calls with task data"]
        direction TB
        P1[Data pipeline]
        P2[Eval harness]
        P3[Training & inference]
        P1 --> P2
        P1 --> P3
        P3 --> P2
    end
    EXT["External APIs<br/>OpenAI / Anthropic / etc."]
    LOCAL -. NEVER .-> EXT

    style EXT fill:#f8d7da,stroke:#900,stroke-width:2px,stroke-dasharray: 5 5,color:#1a1a1a
```

NoteChat is public research data, so this is a *practised discipline* rather
than a legal requirement for this specific corpus — stated plainly in
`PROJECT_SPEC.md` §7a item 3 so it is never misrepresented as a compliance
finding. The harness is built as though the constraint were real, because
demonstrating that correctly is part of the deliverable.

---

## 2. Phase 1 — Data pipeline

```mermaid
flowchart TD
    A["our_revised_v2.csv<br/>columns: data, conversation"] --> B["Validate schema<br/>fail loudly on unexpected columns"]
    B --> C["Drop exact-duplicate notes<br/>by SHA-256 content hash"]
    C --> D["Strip generation-artifact preamble<br/>before first Doctor:/Patient: marker"]
    D --> E["Split 80/10/10 by note_id<br/>seed 42"]
    E --> F[("train / val / test .parquet")]
    E --> G["tests/test_data.py<br/>asserts zero note_id overlap"]
    C --> H["docs/data_report.md<br/>lengths, turn counts, dedup"]
```

Each row is one unique clinical note, so splitting by row *is* splitting by
source document — simpler than the CUAD case, where one contract carried
many annotation rows and row-level splitting would have leaked.

---

## 3. Phase 2 — Eval harness

The harness is the actual deliverable (`PROJECT_SPEC.md` §0). The model is
evidence that it works.

```mermaid
flowchart TD
    IN["One arm's generation"] --> M1["metrics.rouge<br/>ROUGE-1/2/L vs. reference"]
    IN --> M2["metrics.bertscore<br/>semantic similarity vs. reference"]
    IN --> M3["metrics.turn_format_validity<br/>is it a Doctor:/Patient: dialogue?"]
    IN --> M4["faithfulness.numeric_faithfulness<br/>vs. the CLINICAL NOTE"]
    IN --> M5["cost: tok/s, peak VRAM, wall-clock"]

    M1 & M2 & M3 & M4 & M5 --> AGG["metrics.bootstrap_ci<br/>1,000 resamples, 95% percentile"]
    AGG --> OUT[("artifacts/eval/{arm}/results.json")]
    OUT --> CMP["eval.compare<br/>paired_bootstrap_delta across arms"]
    CMP --> CMPOUT[("comparison.json + comparison.md")]

    style M4 fill:#fff3cd,stroke:#b8860b,stroke-width:2px,color:#1a1a1a
```

**Two families of metric, measuring different things.** ROUGE and BERTScore
compare the generation to the *reference dialogue*, which is itself
LLM-generated (§4.2) — they measure similarity to one synthetic exemplar.
`faithfulness.py` (highlighted) instead scores against the *clinical note*,
which is real, and is the only thing in the harness that can detect a fluent
fabrication. Arm 4 exists largely to prove that distinction matters.

`schema.py` and `grounding.py` from the original template were never written:
free-form dialogue has no structured schema to validate and no verbatim
evidence spans to verify (`configs/eval.yaml` records this).

Every metric module is unit-tested against hand-computed examples before it
scores a real model — 46 tests in `tests/`.

---

## 4. Phase 6 — The four arms

```mermaid
flowchart TD
    REC["One held-out clinical note"] --> ARM1 & ARM2 & ARM3 & ARM4

    subgraph ARM1["Arm 1 — floor"]
        A1["Zero-shot Qwen2.5-3B<br/>4-bit"]
    end
    subgraph ARM2["Arm 2 — scale comparison"]
        A2["Zero-shot Qwen2.5-14B<br/>4-bit, ~4.7x params"]
    end
    subgraph ARM3["Arm 3 — the contribution"]
        A3["QLoRA fine-tuned<br/>Qwen2.5-3B"]
    end
    subgraph ARM4["Arm 4 — is an LLM needed?"]
        A4["TF-IDF nearest-neighbour<br/>retrieval, no model"]
    end

    ARM1 & ARM2 & ARM3 & ARM4 --> SCORE["Eval harness"]
    SCORE --> COMPARE["Paired bootstrap:<br/>every pairwise delta + CI"]

    style ARM3 fill:#d3edda,stroke:#3f8f57,stroke-width:2px,color:#1a1a1a
    style ARM4 fill:#fff3cd,stroke:#b8860b,stroke-width:2px,color:#1a1a1a
```

All four arms score the **same** 200 test records, and `compare.py` refuses
to run if that stops being true — a paired bootstrap over differently-sampled
arms would still produce a confident-looking number.

**Arm 4 is not a formality.** A retrieval baseline with no model outscores
the zero-shot 14B on ROUGE-1, because a retrieved dialogue is fluent,
correctly formatted, on-topic — and about the wrong patient. It is the
cleanest demonstration in the project that reference-similarity metrics are
not measuring what a reader assumes they measure. See `README.md` Results.

---

## 5. Phase 3 — Contamination probe

```mermaid
flowchart TD
    D["Held-out reference dialogue"] --> SPLIT2["Split at a turn boundary"]
    SPLIT2 --> PRE["First half"]
    SPLIT2 --> TRUE["True second half"]
    PRE --> BASE["BASE model, raw text<br/>no chat template, greedy"]
    BASE --> GEN["Generated continuation"]
    GEN --> CMP1["ROUGE-L vs. true"]
    GEN --> CMP2["ROUGE-L vs. a RANDOM<br/>other dialogue's second half"]
    CMP1 & CMP2 --> DELTA["Paired delta + CI<br/>gap means memorization"]

    style CMP2 fill:#fff3cd,stroke:#b8860b,stroke-width:2px,color:#1a1a1a
```

The control arm is the whole design. NoteChat dialogues are formulaic, so a
model with zero memorization still scores well above zero against the true
continuation just by writing plausible generic dialogue. Only the *gap*
between true and random is evidence of having seen the text before.

---

## 6. Repo layout

```
company-finetune-eval/
├── configs/              # data.yaml, train.yaml, eval.yaml
├── data/                 # gitignored — raw CSV + processed parquet
├── src/
│   ├── data/build_dataset.py      # Phase 1
│   ├── inference/local_hf.py      # prompts + generation, single source of truth
│   ├── train/train.py             # Phase 5 CLI (imports prompts from local_hf)
│   ├── baselines/classic_baseline.py  # Phase 6 arm 4
│   └── eval/
│       ├── metrics.py             # ROUGE / BERTScore / format / bootstrap
│       ├── faithfulness.py        # numeric grounding vs. the note
│       ├── contamination.py       # Phase 3
│       ├── run_eval.py            # one arm per run
│       └── compare.py             # cross-arm paired deltas
├── scripts/run_all_arms.sh   # reproduces every published number
├── notebooks/finetune.ipynb  # the training run actually used for Phase 5
├── artifacts/
│   ├── adapters/             # LoRA weights (gitignored — large)
│   └── eval/                 # results.json per arm + comparison (committed)
├── tests/                    # 46 tests: data, metrics, baselines, compare
└── docs/
```

---

## 7. What this deliberately does NOT include

- No web UI, API server, or Docker deployment — this is a measurement
  project, judged on rigor rather than deployment surface (`PROJECT_SPEC.md`
  §8).
- No RAG. Note that arm 4 *is* retrieval, but as a baseline to argue
  against, not as a serving architecture.
- No LLM-as-judge metric: grading one model's output with another model
  would reintroduce the synthetic-reference problem §4.2 exists to flag.
- No cloud inference of any kind on task data.
