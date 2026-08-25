"""Phase 2 acceptance tests (PROJECT_SPEC.md §5 Phase 2).

Unit tests against hand-computed toy examples, per the phase's acceptance
criterion. BERTScore is excluded (needs a downloaded roberta-large scorer
model) — exercised instead via run_eval.py's end-to-end acceptance run.
"""

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
