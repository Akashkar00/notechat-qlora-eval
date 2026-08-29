"""Phase 2 acceptance tests (PROJECT_SPEC.md §5 Phase 2).

Unit tests against hand-computed toy examples, per the phase's acceptance
criterion. BERTScore is excluded (needs a downloaded roberta-large scorer
model) — exercised instead via run_eval.py's end-to-end acceptance run.
"""

from src.eval.faithfulness import extract_numbers, numeric_faithfulness
from src.eval.metrics import bootstrap_ci, paired_bootstrap_delta, rouge, turn_format_validity


def test_rouge_identical_strings_score_one():
    text = "Doctor: how are you feeling. Patient: not great"
    scores = rouge(text, text)
    assert scores["rouge1"] == 1.0
    assert scores["rouge2"] == 1.0
    assert scores["rougeL"] == 1.0


def test_rouge_disjoint_strings_score_zero():
    scores = rouge("apple banana cherry", "xylophone quibble zephyr")
    assert scores["rouge1"] == 0.0
    assert scores["rouge2"] == 0.0
    assert scores["rougeL"] == 0.0


def test_rouge_partial_overlap_hand_computed():
    # reference has 4 unigrams, generated has 4 unigrams, 2 in common
    # -> precision = recall = 2/4 = 0.5 -> f1 = 0.5
    scores = rouge(generated="the cat sat down", reference="the cat ran away")
    assert scores["rouge1"] == 0.5


def test_turn_format_validity_well_formed_dialogue():
    text = "Doctor: Hello.\nPatient: Hi doctor.\nDoctor: How do you feel?"
    result = turn_format_validity(text)
    assert result["starts_with_turn_marker"] is True
    assert result["num_turns"] == 3
    assert result["has_doctor_turn"] is True
    assert result["has_patient_turn"] is True


def test_turn_format_validity_no_markers_at_all():
    result = turn_format_validity("just some free-form prose with no dialogue structure")
    assert result["starts_with_turn_marker"] is False
    assert result["num_turns"] == 0
    assert result["has_doctor_turn"] is False
    assert result["has_patient_turn"] is False


def test_turn_format_validity_leading_preamble_before_marker():
    # split_turns (reused from build_dataset.py) only splits at turn
    # markers, it doesn't strip a non-turn prefix — that's strip_preamble's
    # job, applied upstream at build time, not here. So the preamble text
    # itself counts as one (non-Doctor/Patient) "turn" segment.
    text = "some leaked prompt text\nDoctor: Hello.\nPatient: Hi."
    result = turn_format_validity(text)
    assert result["starts_with_turn_marker"] is False
    assert result["num_turns"] == 3
    assert result["has_doctor_turn"] is True
    assert result["has_patient_turn"] is True


def test_bootstrap_ci_constant_values_has_zero_width_ci():
    result = bootstrap_ci([5.0, 5.0, 5.0, 5.0], n_resamples=100, seed=0)
    assert result["point_estimate"] == 5.0
    assert result["ci_low"] == 5.0
    assert result["ci_high"] == 5.0
    assert result["n"] == 4


def test_bootstrap_ci_point_estimate_matches_mean():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = bootstrap_ci(values, n_resamples=500, seed=1)
    assert abs(result["point_estimate"] - 0.3) < 1e-9
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]


def test_bootstrap_ci_is_deterministic_given_seed():
    values = [0.1, 0.9, 0.4, 0.2, 0.7]
    a = bootstrap_ci(values, n_resamples=200, seed=42)
    b = bootstrap_ci(values, n_resamples=200, seed=42)
    assert a == b


def test_paired_bootstrap_delta_identical_arrays_is_zero():
    values = [0.1, 0.2, 0.3, 0.4]
    result = paired_bootstrap_delta(values, values, n_resamples=200, seed=0)
    assert result["delta"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0


def test_paired_bootstrap_delta_hand_computed_constant_offset():
    a = [0.5, 0.6, 0.7, 0.8]
    b = [0.4, 0.5, 0.6, 0.7]  # every record is exactly 0.1 higher in a
    result = paired_bootstrap_delta(a, b, n_resamples=200, seed=0)
    assert abs(result["delta"] - 0.1) < 1e-9
    assert abs(result["ci_low"] - 0.1) < 1e-9
    assert abs(result["ci_high"] - 0.1) < 1e-9


def test_paired_bootstrap_delta_rejects_mismatched_lengths():
    import pytest

    with pytest.raises(ValueError):
        paired_bootstrap_delta([0.1, 0.2], [0.1], n_resamples=10)


# --- Faithfulness proxy (src/eval/faithfulness.py) ---


def test_extract_numbers_finds_integers_and_decimals():
    assert extract_numbers("BP 135/85, temp 37.2, age 64") == {"135", "85", "37.2", "64"}


def test_extract_numbers_normalizes_trailing_zero_decimals():
    # 5.0 and 5 are the same clinical claim written two ways; they must not
    # count as a match failure.
    assert extract_numbers("dose 5.0 mg") == extract_numbers("dose 5 mg") == {"5"}


def test_extract_numbers_splits_compound_values():
    # "135/85" is two separately-checkable claims, and "64-year-old" carries
    # a real number that should be verifiable.
    assert extract_numbers("a 64-year-old with BP 135/85") == {"64", "135", "85"}


def test_numeric_faithfulness_perfect_when_all_numbers_carried_over():
    note = "A 64-year-old man. Blood pressure was 135/85 mmHg."
    gen = "Doctor: You're 64, and your blood pressure was 135/85."
    result = numeric_faithfulness(gen, note)
    assert result["numeric_grounding_recall"] == 1.0
    assert result["numeric_precision"] == 1.0
    assert result["fabricated_number_rate"] == 0.0


def test_numeric_faithfulness_detects_fabricated_values():
    note = "A 64-year-old man. Blood pressure was 135/85 mmHg."
    gen = "Doctor: You're 70, and your blood pressure was 200/110."
    result = numeric_faithfulness(gen, note)
    # Nothing shared: note {64,135,85} vs generation {70,200,110}
    assert result["numeric_grounding_recall"] == 0.0
    assert result["numeric_precision"] == 0.0
    assert result["fabricated_number_rate"] == 1.0


def test_numeric_faithfulness_hand_computed_partial_overlap():
    note = "Age 50. Pulse 80. Temp 37."  # {50, 80, 37}
    gen = "Doctor: you are 50 and your pulse is 80, weight 99."  # {50, 80, 99}
    result = numeric_faithfulness(gen, note)
    assert result["numeric_grounding_recall"] == 2 / 3  # 50, 80 carried over; 37 missed
    assert result["numeric_precision"] == 2 / 3  # 99 fabricated
    assert abs(result["fabricated_number_rate"] - 1 / 3) < 1e-9


def test_numeric_faithfulness_generation_with_no_numbers_fabricates_nothing():
    result = numeric_faithfulness("Doctor: How are you? Patient: Fine.", "A 64-year-old man.")
    assert result["numeric_precision"] == 1.0  # stated nothing, so invented nothing
    assert result["fabricated_number_rate"] == 0.0
    assert result["numeric_grounding_recall"] == 0.0  # but carried none of the note's specifics


def test_numeric_faithfulness_reports_counts():
    result = numeric_faithfulness("pulse 80", "Age 50. Pulse 80.")
    assert result["n_note_numbers"] == 2
    assert result["n_generated_numbers"] == 1
