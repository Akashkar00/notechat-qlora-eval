"""Phase 3 contamination-probe tests (PROJECT_SPEC.md §5 Phase 3).

Only the non-GPU logic is unit-tested here; the probe itself needs a loaded
base model and is exercised by `python -m src.eval.contamination`.
"""

import random

import pytest

from src.eval.contamination import derange, split_dialogue_halves


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


def test_derange_has_no_fixed_points():
    items = [f"continuation-{i}" for i in range(50)]
    deranged = derange(items, random.Random(0))
    assert len(deranged) == len(items)
    assert sorted(deranged) == sorted(items)  # a permutation, nothing lost or duplicated
    assert all(a != b for a, b in zip(items, deranged, strict=True))


def test_derange_no_fixed_points_across_many_seeds():
    # A plain rng.shuffle leaves at least one fixed point ~63% of the time at
    # this size, which is the bug this function exists to prevent — so the
    # guarantee has to hold for every seed, not just a lucky one.
    items = [f"continuation-{i}" for i in range(100)]
    for seed in range(25):
        deranged = derange(items, random.Random(seed))
        assert all(a != b for a, b in zip(items, deranged, strict=True)), f"fixed point at seed {seed}"


def test_derange_is_deterministic_given_seed():
    items = [f"continuation-{i}" for i in range(20)]
    assert derange(items, random.Random(42)) == derange(items, random.Random(42))


def test_derange_rejects_too_few_records():
    with pytest.raises(ValueError, match="at least 2 records"):
        derange(["only-one"], random.Random(0))
