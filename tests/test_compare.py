"""Tests for the cross-arm comparison entrypoint (src/eval/compare.py).

The guard rail here matters more than the arithmetic: a paired bootstrap
over arms that were scored on *different* records would still produce a
confident-looking delta. That must fail loudly, not silently intersect.
"""

import pytest

from src.eval.compare import assert_comparable, compare_pair

BS_CFG = {"n_resamples": 200, "confidence_level": 0.95}


def _arm(name: str, note_ids: list[str], rouge1_values: list[float], do_sample: bool = False) -> dict:
    return {
        "arm": name,
        "decoding": {"do_sample": do_sample},
        "records": [
            {"note_id": nid, "rouge1": v, "rouge2": v, "rougeL": v, "bertscore_f1": v}
            for nid, v in zip(note_ids, rouge1_values)
        ],
    }


def test_assert_comparable_accepts_identical_record_sets():
    arms = {
        "a": _arm("a", ["n1", "n2", "n3"], [0.5, 0.6, 0.7]),
        "b": _arm("b", ["n1", "n2", "n3"], [0.4, 0.5, 0.6]),
    }
    assert assert_comparable(arms) == ["n1", "n2", "n3"]


def test_assert_comparable_rejects_mismatched_record_sets():
    arms = {
        "a": _arm("a", ["n1", "n2", "n3"], [0.5, 0.6, 0.7]),
        "b": _arm("b", ["n1", "n2", "n9"], [0.4, 0.5, 0.6]),
    }
    with pytest.raises(ValueError, match="not scored on the same records"):
        assert_comparable(arms)


def test_assert_comparable_rejects_different_record_counts():
    arms = {
        "a": _arm("a", ["n1", "n2", "n3"], [0.5, 0.6, 0.7]),
        "b": _arm("b", ["n1", "n2"], [0.4, 0.5]),
    }
    with pytest.raises(ValueError):
        assert_comparable(arms)


def test_assert_comparable_warns_on_stochastic_decoding(capsys):
    arms = {
        "a": _arm("a", ["n1", "n2"], [0.5, 0.6], do_sample=True),
        "b": _arm("b", ["n1", "n2"], [0.4, 0.5]),
    }
    assert_comparable(arms)
    assert "stochastic decoding" in capsys.readouterr().out


def test_compare_pair_constant_offset_recovers_exact_delta():
    arms = {
        "a": _arm("a", ["n1", "n2", "n3"], [0.5, 0.6, 0.7]),
        "b": _arm("b", ["n1", "n2", "n3"], [0.4, 0.5, 0.6]),  # exactly 0.1 lower everywhere
    }
    deltas = compare_pair(arms, "a", "b", BS_CFG)
    assert abs(deltas["rouge1"]["delta"] - 0.1) < 1e-9
    assert deltas["rouge1"]["excludes_zero"] is True


def test_compare_pair_identical_arms_delta_is_zero_and_includes_zero():
    arms = {
        "a": _arm("a", ["n1", "n2", "n3"], [0.5, 0.6, 0.7]),
        "b": _arm("b", ["n1", "n2", "n3"], [0.5, 0.6, 0.7]),
    }
    deltas = compare_pair(arms, "a", "b", BS_CFG)
    assert deltas["rouge1"]["delta"] == 0.0
    assert deltas["rouge1"]["excludes_zero"] is False


def test_compare_pair_is_order_sensitive():
    arms = {
        "a": _arm("a", ["n1", "n2"], [0.5, 0.6]),
        "b": _arm("b", ["n1", "n2"], [0.4, 0.5]),
    }
    forward = compare_pair(arms, "a", "b", BS_CFG)["rouge1"]["delta"]
    backward = compare_pair(arms, "b", "a", BS_CFG)["rouge1"]["delta"]
    assert abs(forward + backward) < 1e-9


def test_compare_pair_skips_metrics_an_arm_lacks():
    """An older results.json predating a metric shouldn't crash the compare."""
    arms = {
        "a": _arm("a", ["n1", "n2"], [0.5, 0.6]),
        "b": _arm("b", ["n1", "n2"], [0.4, 0.5]),
    }
    # Neither toy arm records the faithfulness metrics.
    deltas = compare_pair(arms, "a", "b", BS_CFG)
    assert "numeric_grounding_recall" not in deltas
    assert "rouge1" in deltas
