"""Faithfulness proxy: numeric grounding (PROJECT_SPEC.md §5 Phase 2).

Every other metric in this harness compares the generation to a *reference
dialogue*. That reference is itself LLM-generated (§4.2), so all of them are
blind to the failure mode found during the Phase 5 sanity check: a fluent,
correctly-formatted dialogue that states things the source note never said.
This module measures that failure mode directly against the **clinical
note** — the one artifact in this dataset that is real.

**Method.** Extract every number from the note and from the generation, then:

- `numeric_grounding_recall` — of the note's numbers, how many appear in the
  generated dialogue. Low means the dialogue ignored the note's specifics.
- `numeric_precision` — of the numbers the dialogue states, how many are
  supported by the note. Low means the dialogue invented clinical values.
- `fabricated_number_rate` — 1 − precision, stated directly because "made up
  a lab value" is the number a clinician would actually ask about.

**Why numbers.** Clinical notes are number-dense (age, blood pressure, heart
rate, lab values, dosages, durations) and a numeric claim is unambiguously
checkable, unlike paraphrased prose. It is a *proxy*: it cannot catch a
fabricated diagnosis stated without a number, and it will count a correct
value that the model reworded ("about two weeks" for "14 days") as missing.
It is deliberately a cheap, deterministic, explainable check rather than an
LLM-as-judge, which would reintroduce exactly the "grade one model's output
with another model" problem §4.2 warns about.

Read it as a floor on fabrication, not a certificate of clinical accuracy.
"""

import re

# Integers and decimals. Deliberately splits "135/85" into 135 and 85, and
# pulls 64 out of "64-year-old" — both are real, separately-checkable claims.
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def extract_numbers(text: str) -> set[str]:
    """Numbers in `text`, normalized so 5, 5.0 and 05 compare equal.

    Returns a set, not a multiset: the question is "was this value stated,"
    not "how many times." Repeating the patient's age three times in a
    dialogue is normal conversational behaviour, not three separate claims.
    """
    normalized = set()
    for raw in NUMBER_RE.findall(text):
        value = float(raw)
        # Render 5.0 as "5" so it matches an integer 5 written elsewhere.
        normalized.add(str(int(value)) if value.is_integer() else str(value))
    return normalized


def numeric_faithfulness(generated: str, clinical_note: str) -> dict[str, float]:
    """Numeric grounding of one generation against its source note.

    Vacuous cases resolve to 1.0 rather than being dropped, so every record
    yields a score and the arrays stay index-aligned for
    `metrics.bootstrap_ci` / `metrics.paired_bootstrap_delta`:
    a generation stating no numbers has fabricated nothing (precision 1.0),
    and a note containing no numbers has no specifics to miss (recall 1.0).
    Clinical notes in this corpus essentially always contain numbers, so the
    recall case is theoretical; the precision case is not (a model can emit a
    number-free dialogue).
    """
    note_numbers = extract_numbers(clinical_note)
    gen_numbers = extract_numbers(generated)
    shared = note_numbers & gen_numbers

    recall = len(shared) / len(note_numbers) if note_numbers else 1.0
    precision = len(shared) / len(gen_numbers) if gen_numbers else 1.0

    return {
        "numeric_grounding_recall": recall,
        "numeric_precision": precision,
        "fabricated_number_rate": 1.0 - precision,
        "n_note_numbers": len(note_numbers),
        "n_generated_numbers": len(gen_numbers),
    }
