"""Phase 2 — generation metrics for the NoteChat dialogue task (PROJECT_SPEC.md §5 Phase 2).

`src/eval/schema.py` and `src/eval/grounding.py` from the repo-structure
template don't apply here — the output is free-form dialogue, not a fixed
schema with verbatim evidence spans (see PROJECT_SPEC.md §4). This module
covers what does apply: reference-based similarity (ROUGE, BERTScore) and a
structural format check, both with bootstrap CIs.

Reference-based metrics measure similarity to the `conversation` column,
which is itself LLM-generated, not ground truth (PROJECT_SPEC.md §4.2) —
report alongside that caveat, never as correctness.
"""

import numpy as np
from rouge_score import rouge_scorer

from src.data.build_dataset import TURN_START_RE, split_turns

_ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def rouge(generated: str, reference: str) -> dict[str, float]:
    """F-measure for rouge1/rouge2/rougeL. Argument order matches
    rouge_score's own convention: score(target, prediction)."""
    scores = _ROUGE_SCORER.score(reference, generated)
    return {name: result.fmeasure for name, result in scores.items()}


def bertscore(generated: list[str], reference: list[str]) -> dict[str, list[float]]:
    """Batched BERTScore P/R/F1 (one triple per generated/reference pair).

    Imported lazily, and shipped in the `gpu` extra rather than the base
    dependency group: it pulls in torch and downloads a roberta-large
    scorer. Keeping it out of module scope is what lets the metric unit
    tests (and CI) run on a machine with no CUDA stack at all.

    **Raw, not baseline-rescaled** (`rescale_with_baseline=False`, the
    default). Raw BERTScore has a high, task-dependent floor — two unrelated
    English texts still score ~0.80 — so absolute values compress into a
    narrow band near the top of the range and a 0.85-vs-0.91 gap looks far
    smaller than it is. Rescaling against BERTScore's baseline would spread
    that band out and make the numbers read more intuitively.

    It is deliberately left off because the rescaling baseline is a fixed
    constant per (model, language): subtracting it is a monotone affine
    transform of every score, so it cannot change the sign of any paired
    delta, which arm is ranked above which, or whether a bootstrap CI
    excludes zero. It would only rescale figures that are always reported
    against each other rather than against an external threshold. Every
    absolute BERTScore in this repo must therefore be read comparatively —
    "arm 3 above arm 2", never "0.910 is 91% correct".
    """
    try:
        from bert_score import score
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "bert-score is not installed. It ships in the `gpu` extra "
            "(`uv sync --extra gpu`) because it depends on torch."
        ) from exc

    precision, recall, f1 = score(generated, reference, lang="en", verbose=False)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }


def turn_format_validity(generated: str) -> dict[str, bool | int]:
    """Structural check independent of reference similarity: does the raw
    generation actually parse as a Doctor:/Patient: dialogue? This is the
    closest analogue this task has to schema.py's "% valid structured
    output" (PROJECT_SPEC.md §5 Phase 2), since there's no JSON schema to
    validate. Not a strict-alternation check — the real data itself doesn't
    strictly alternate (docs/data_report.md: doctor/patient turn counts per
    conversation aren't equal), so alternation would flag valid dialogues.
    """
    stripped = generated.strip()
    has_marker = bool(TURN_START_RE.search(stripped))
    turns = split_turns(stripped) if has_marker else []
    return {
        "starts_with_turn_marker": bool(TURN_START_RE.match(stripped)),
        "num_turns": len(turns),
        "has_doctor_turn": any(t.startswith("Doctor:") for t in turns),
        "has_patient_turn": any(t.startswith("Patient:") for t in turns),
    }


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Record-level percentile bootstrap CI on the mean (PROJECT_SPEC.md §5
    Phase 2: "1000 resamples at the record level, 95% percentile CI")."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    resample_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resample_means[i] = arr[idx].mean()
    alpha = 1 - confidence_level
    lo, hi = np.percentile(resample_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point_estimate": float(arr.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": n,
        "n_resamples": n_resamples,
    }


def paired_bootstrap_delta(
    values_a: list[float],
    values_b: list[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """CI on the mean paired difference (a - b) for model-A-vs-model-B
    comparisons over the same records (PROJECT_SPEC.md §5 Phase 2: "Paired
    bootstrap ... Report the delta CI.")."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"paired bootstrap requires equal-length, index-aligned arrays, got {len(a)} vs {len(b)}")
    diffs = a - b
    rng = np.random.default_rng(seed)
    n = len(diffs)
    resample_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resample_means[i] = diffs[idx].mean()
    alpha = 1 - confidence_level
    lo, hi = np.percentile(resample_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "delta": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": n,
        "n_resamples": n_resamples,
    }
