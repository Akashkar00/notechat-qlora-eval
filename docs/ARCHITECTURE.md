# Architecture — On-Prem Fine-Tuning on Company Data

Full system architecture for the project defined in `PROJECT_SPEC.md`. This
document is a placeholder cloned from `clinical-coding-eval`'s
`ARCHITECTURE.md` — rewrite the diagrams below once `OPEN_DECISIONS` in
`PROJECT_SPEC.md` is filled in and the task is concrete. The one part that
should survive unchanged regardless of task: **everything stays local**
unless `OPEN_DECISIONS` #3 explicitly permits otherwise.

---

## 1. Big picture — how data becomes a result

```mermaid
flowchart TD
    A[Company data source\nsee OPEN_DECISIONS #2] --> B[Data Pipeline\nPhase 1]
    B --> C[(Clean dataset\ntrain / val / test\nsplit on the correct key)]

    C --> D[Eval Harness\nPhase 2]
    C --> E[Four Model Arms\nPhase 5 + 6]

    E --> D
    D --> F[Results:\nmetrics + confidence intervals]
    F --> G[Write-up\nREADME, MODEL_CARD\nPhase 7]

    style A fill:#f8d7da,stroke:#b4545a,stroke-width:2px,color:#1a1a1a
    style G fill:#d3edda,stroke:#3f8f57,stroke-width:2px,color:#1a1a1a
```

**Everything inside the dashed boundary below runs on the local machine
only, unless `OPEN_DECISIONS` #3 explicitly permits an external API for a
named, non-sensitive subset of the data.**

```mermaid
flowchart LR
    subgraph LOCAL["This computer only — no network calls with company data"]
        direction TB
        P1[Data Pipeline]
        P2[Eval Harness]
        P3[Model Inference\n& Training]
        P1 --> P2
        P1 --> P3
        P3 --> P2
    end
    EXT[External APIs\nOpenAI / Anthropic / etc.]
    LOCAL -. NEVER\nunless OPEN_DECISIONS #3\nsays otherwise .-> EXT

    style EXT fill:#f8d7da,stroke:#900,stroke-width:2px,stroke-dasharray: 5 5,color:#1a1a1a
```

---

## 2. Phase 1 — Data pipeline (placeholder, task-dependent)

```mermaid
flowchart TD
    A1[Raw company data\nsee OPEN_DECISIONS #2] --> B1[Filter to\nin-scope population]
    B1 --> C1[Derive label / output\nspace, see PROJECT_SPEC §4.1]
    C1 --> D1[Deduplicate\nif applicable]
    D1 --> E1[Split on the correct\ngrouping key, OPEN_DECISIONS #5]
    E1 --> F1[(train.parquet\nval.parquet\ntest.parquet)]

    E1 --> G1[test_data.py\nasserts zero overlap\nacross splits on the\ngrouping key]
    D1 --> H1[docs/data_report.md\ncoverage, distribution,\ndedup count]
```

**Why splitting on the right key matters:** if the same grouping unit
(customer, document source, whatever `OPEN_DECISIONS` #5 names) appears in
both train and test, the model can cheat by memorizing that unit instead of
learning the task.

---

## 3. Phase 2 — Evaluation harness (built and tested *before* any model runs)

```mermaid
flowchart TD
    IN[Model output] --> M1[schema.py\nvalid structured output?\nif applicable]
    IN --> M2[grounding.py\nis any evidence/citation\nverifiable? if applicable]
    IN --> M3[metrics.py\ntask metric(s)\n+ bootstrap confidence intervals]
    IN --> M4[cost / latency\ntokens per second,\nVRAM used, time per record]

    M1 & M2 & M3 & M4 --> OUT[run_eval.py\none combined results.json\nper model arm]
```

Each metric module has its own unit tests, checked against hand-computed
examples first, so the scoring code is trusted before it ever touches a
real model.

---

## 4. Phase 5 & 6 — The four things being compared

```mermaid
flowchart TD
    REC[Company data record] --> ARM1
    REC --> ARM2
    REC --> ARM3
    REC --> ARM4

    subgraph ARM1["Arm 1 — Zero-shot small model"]
        A1[Small model, untrained\nfloor / baseline]
    end
    subgraph ARM2["Arm 2 — Zero-shot large baseline"]
        A2[Larger open-weight model\nsee OPEN_DECISIONS #7]
    end
    subgraph ARM3["Arm 3 — Fine-tuned small model — the contribution"]
        A3[Small model\n+ QLoRA fine-tune]
    end
    subgraph ARM4["Arm 4 — Classic approach"]
        A4[Non-LLM baseline\nencoder / rules / regex\ntask-dependent]
    end

    ARM1 --> SCORE[Eval Harness]
    ARM2 --> SCORE
    ARM3 --> SCORE
    ARM4 --> SCORE
    SCORE --> COMPARE[Paired bootstrap:\nis Arm 3 really better\nthan Arm 1 / 2 / 4?\nreport the delta + CI]
```

**The honest possible outcomes, all of which are valid findings:**
- Arm 3 (fine-tuned small model) matches or beats Arm 2 (large model) → the
  thesis holds.
- Arm 4 (classic approach) beats everything → "the right tool for this task
  isn't a generative LLM at all," reported prominently, not buried.

---

## 5. Code / repo layout

```
company-finetune-eval/
├── configs/                 # YAML configs: data, training, eval
├── src/
│   ├── data/                 # Phase 1 — pipeline
│   ├── train/                 # Phase 5 — QLoRA fine-tuning
│   ├── inference/             # Runs any of the 4 arms locally
│   ├── baselines/              # Arm 4 — classic approach
│   └── eval/                    # Phase 2 — the harness (section 3 above)
├── annotation/                  # Phase 4 — guidelines, sampling, kappa (if needed)
├── tests/                        # pytest — leakage + metric unit tests
└── docs/
    ├── DECISIONS.md               # running log: what was chosen and why
    ├── data_report.md
    ├── contamination_report.md
    ├── annotation_report.md
    └── MODEL_CARD.md
```

---

## 6. What this architecture deliberately does NOT include

- No web UI, no API server, no Docker deployment — this is a
  research/measurement project, not a product.
- No RAG, no external retrieval, unless the task genuinely requires it and
  `OPEN_DECISIONS` says so.
- No cloud inference of any kind on sensitive company data, unless
  `OPEN_DECISIONS` #3 explicitly permits it for a named subset.
