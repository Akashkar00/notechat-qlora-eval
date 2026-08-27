"""Phase 3 contamination-probe tests (PROJECT_SPEC.md §5 Phase 3).

Only the non-GPU logic is unit-tested here; the probe itself needs a loaded
base model and is exercised by `python -m src.eval.contamination`.
"""

from src.eval.contamination import split_dialogue_halves


def test_splits_at_a_turn_boundary_not_mid_sentence():
    conv = "Doctor: A.\nPatient: B.\nDoctor: C.\nPatient: D."
    prefix, continuation = split_dialogue_halves(conv)
    # Both halves must start with a speaker marker — a mid-turn split would
    # measure sentence completion rather than corpus memory.
    assert prefix.startswith("Doctor:")
    assert continuation.startswith("Doctor:") or continuation.startswith("Patient:")


def test_halves_reconstruct_the_original_turns():
    conv = "Doctor: A.\nPatient: B.\nDoctor: C.\nPatient: D.\nDoctor: E.\nPatient: F."
    prefix, continuation = split_dialogue_halves(conv)
    assert prefix.split("\n") + continuation.split("\n") == conv.split("\n")


def test_split_is_balanced():
    conv = "\n".join(f"Doctor: turn {i}." for i in range(8))
    prefix, continuation = split_dialogue_halves(conv)
    assert len(prefix.split("\n")) == len(continuation.split("\n")) == 4


def test_too_short_to_split_returns_none():
    # Fewer than 4 turns leaves a prefix too small to probe meaningfully.
    assert split_dialogue_halves("Doctor: A.\nPatient: B.") is None


def test_no_turn_markers_returns_none():
    assert split_dialogue_halves("just prose with no dialogue structure at all") is None
