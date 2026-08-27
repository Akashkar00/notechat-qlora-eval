"""Phase 3 — contamination / memorization probe (PROJECT_SPEC.md §5 Phase 3).

**The question.** NoteChat is a public dataset and its clinical notes come
from PMC-Patients (public case reports on PubMed Central). If the base model
already memorized this corpus during pretraining, then "fine-tuning improved
the scores" could partly mean "fine-tuning surfaced text the model already
had," which is a materially weaker claim than "fine-tuning taught the model
the task." §5 Phase 3 requires testing this before claiming the latter.

**The method — prefix continuation.** Take a reference dialogue, feed the
**base** model its first half as raw text (no chat template: we are probing
the language model's memory, not its instruction-following), let it
continue, and measure ROUGE-L between what it produced and the true second
half.

**Why a control is essential.** NoteChat dialogues are formulaic — a doctor
restating a note's facts as questions — so *any* competent model scores well
above zero on this by writing generic plausible dialogue, with no
memorization at all. Scoring the same generated continuation against a
*different, randomly chosen* dialogue's second half gives the "generic
style" floor. The comparison that matters is:

    overlap(generated, true continuation)  vs  overlap(generated, random continuation)

If those are indistinguishable, the model is producing style, not recall,
and there is no evidence of contamination. A large gap in favour of the true
continuation is evidence the model has seen this specific text before.

Reported as a paired bootstrap delta with a CI, like every other comparison
in this repo, so "no evidence of contamination" is a measured statement
rather than an assertion.

Run:
    python -m src.eval.contamination
    python -m src.eval.contamination --n-records 100 --split test
"""

import argparse
import json
import random
from pathlib import Path

import polars as pl
import yaml

from src.data.build_dataset import split_turns
from src.eval.metrics import bootstrap_ci, paired_bootstrap_delta, rouge

TRAIN_CONFIG_PATH = Path("configs/train.yaml")
EVAL_CONFIG_PATH = Path("configs/eval.yaml")
PROCESSED_DIR = Path("data/processed")
REPORT_PATH = Path("docs/contamination_report.md")
RESULTS_PATH = Path("artifacts/eval/contamination.json")


def split_dialogue_halves(conversation: str) -> tuple[str, str] | None:
    """Split a dialogue into (prefix, true continuation) at a turn boundary.

    Splitting mid-turn would hand the model a truncated sentence and measure
    its ability to finish a sentence rather than its memory of the corpus.
    Returns None for dialogues too short to split meaningfully.
    """
    turns = split_turns(conversation)
    if len(turns) < 4:
        return None
    midpoint = len(turns) // 2
    return "\n".join(turns[:midpoint]), "\n".join(turns[midpoint:])


def probe(model_name: str, n_records: int, split: str, seed: int, max_new_tokens: int = 256) -> dict:
    train_cfg = yaml.safe_load(TRAIN_CONFIG_PATH.read_text())
    eval_cfg = yaml.safe_load(EVAL_CONFIG_PATH.read_text())

    import torch

    from src.inference.local_hf import load_model

    parquet_path = PROCESSED_DIR / f"{split}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} not found — run `python -m src.data.build_dataset` first.")
    df = pl.read_parquet(parquet_path)
    if n_records < len(df):
        df = df.sample(n=n_records, seed=seed)

    pairs = []
    for row in df.iter_rows(named=True):
        halves = split_dialogue_halves(row["conversation"])
        if halves is not None:
            pairs.append((row["note_id"], *halves))

    # Deterministic shuffle for the control: each record's generated
    # continuation is scored against some *other* record's true continuation.
    rng = random.Random(seed)
    shuffled = [p[2] for p in pairs]
    rng.shuffle(shuffled)

    # Base model only. Probing the fine-tuned model would conflate
    # pretraining memorization (the question) with our own fine-tune having
    # trained on this corpus (not the question).
    model, tokenizer = load_model(
        model_name=model_name,
        max_seq_len=train_cfg["max_seq_len"],
        load_in_4bit=train_cfg["quantization"]["load_in_4bit"],
        seed=train_cfg["seed"],
        adapter_path=None,
    )

    records = []
    for (note_id, prefix, true_continuation), control_continuation in zip(pairs, shuffled):
        # Raw continuation, no chat template — this probes the LM's memory of
        # the text itself, not its behaviour as an assistant.
        inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy: memorized text is the argmax path
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        records.append(
            {
                "note_id": note_id,
                "generated_continuation": generated,
                "rougeL_vs_true": rouge(generated, true_continuation)["rougeL"],
                "rougeL_vs_control": rouge(generated, control_continuation)["rougeL"],
            }
        )

    bs_cfg = eval_cfg["bootstrap"]
    true_scores = [r["rougeL_vs_true"] for r in records]
    control_scores = [r["rougeL_vs_control"] for r in records]
    delta = paired_bootstrap_delta(
        true_scores, control_scores, n_resamples=bs_cfg["n_resamples"], confidence_level=bs_cfg["confidence_level"]
    )
    delta["excludes_zero"] = delta["ci_low"] > 0 or delta["ci_high"] < 0

    return {
        "model_name": model_name,
        "split": split,
        "n_records": len(records),
        "rougeL_vs_true": bootstrap_ci(true_scores, **bs_cfg),
        "rougeL_vs_control": bootstrap_ci(control_scores, **bs_cfg),
        "delta_true_minus_control": delta,
        "records": records,
    }


def write_report(result: dict) -> None:
    true_ci = result["rougeL_vs_true"]
    ctrl_ci = result["rougeL_vs_control"]
    delta = result["delta_true_minus_control"]
    contaminated = delta["excludes_zero"] and delta["delta"] > 0

    verdict = (
        "**Evidence of memorization.** The base model continues held-out NoteChat "
        "dialogues measurably closer to their true continuation than to an unrelated "
        "one. Fine-tuning gains on this corpus should be read with that in mind."
        if contaminated
        else "**No evidence of memorization.** The base model's continuations are no closer "
        "to the true continuation than to an unrelated dialogue from the same corpus — "
        "i.e. it produces NoteChat-*style* text without recalling these specific "
        "dialogues. The fine-tuning gains reported in README.md are therefore not "
        "explained by pretraining contamination."
    )

    lines = [
        "# Contamination report (Phase 3)",
        "",
        "Generated by `python -m src.eval.contamination`. Do not hand-edit.",
        "",
        "## Why this check exists",
        "",
        "NoteChat is public and its clinical notes come from PMC-Patients (public",
        "PubMed Central case reports), so the base model may have seen this corpus",
        "during pretraining. If it had memorized it, the fine-tuning improvements in",
        "`README.md` would partly reflect recall rather than learning.",
        "",
        "## Method",
        "",
        f"- Model probed: `{result['model_name']}` (**base**, no adapter — this asks",
        "  what pretraining knew, not what our fine-tune taught).",
        f"- {result['n_records']} dialogues from the `{result['split']}` split, each split in half",
        "  at a turn boundary.",
        "- The first half is fed as **raw text** (no chat template) and continued with",
        "  greedy decoding; memorized text is the argmax path.",
        "- **Control:** the same generated continuation is also scored against a",
        "  *different* dialogue's second half. NoteChat dialogues are formulaic, so a",
        "  non-memorizing model still scores well above zero — the control is what",
        "  separates 'knows the style' from 'knows this text'.",
        "",
        "## Results (ROUGE-L, 95% bootstrap CI)",
        "",
        "| Comparison | ROUGE-L | 95% CI |",
        "|---|---|---|",
        f"| Generated vs. **true** continuation | {true_ci['point_estimate']:.4f} | "
        f"[{true_ci['ci_low']:.4f}, {true_ci['ci_high']:.4f}] |",
        f"| Generated vs. **random** continuation (control) | {ctrl_ci['point_estimate']:.4f} | "
        f"[{ctrl_ci['ci_low']:.4f}, {ctrl_ci['ci_high']:.4f}] |",
        "",
        "### Paired delta (true − control)",
        "",
        "| Δ ROUGE-L | 95% CI | Excludes zero |",
        "|---|---|---|",
        f"| {delta['delta']:+.4f} | [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}] | "
        f"{'yes' if delta['excludes_zero'] else 'no'} |",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        "## Limitations",
        "",
        "- Prefix continuation detects *verbatim* memorization. A model that absorbed",
        "  the corpus's content without reproducing its wording would not be caught.",
        "- Only the base model is probed. This does not measure whether our own",
        "  fine-tune overfit its training split — a different question, answerable by",
        "  comparing train-split and test-split scores in `artifacts/eval/`.",
        "- ROUGE-L is a surface-overlap measure; a paraphrased recall would score low.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    parser.add_argument("--n-records", type=int, default=100)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = probe(args.model_name, args.n_records, args.split, args.seed)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(result)
    print(f"Wrote {RESULTS_PATH} and {REPORT_PATH}")
    print(
        f"ROUGE-L vs true: {result['rougeL_vs_true']['point_estimate']:.4f} | "
        f"vs control: {result['rougeL_vs_control']['point_estimate']:.4f} | "
        f"delta {result['delta_true_minus_control']['delta']:+.4f} "
        f"(excludes zero: {result['delta_true_minus_control']['excludes_zero']})"
    )


if __name__ == "__main__":
    main()
