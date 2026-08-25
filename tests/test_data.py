"""Phase 1 acceptance tests (PROJECT_SPEC.md §5 Phase 1).

Runs against the real processed parquet files, which is why the no-leakage
assertion here is the actual acceptance test — not a synthetic stand-in.
"""

from pathlib import Path

import polars as pl
import pytest

from src.data.build_dataset import build_records, load_raw, split_records, split_turns, strip_preamble

PROCESSED_DIR = Path("data/processed")


def _skip_if_not_built():
    if not (PROCESSED_DIR / "train.parquet").exists():
        pytest.skip("data/processed/*.parquet not built yet — run `python -m src.data.build_dataset` first")


def test_strip_preamble_removes_leading_text():
    conv = "some leaked keyword list\nDoctor: Hello, how are you?\nPatient: Fine."
    cleaned, n = strip_preamble(conv)
    assert cleaned == "Doctor: Hello, how are you?\nPatient: Fine."
    assert n == len("some leaked keyword list\n")


def test_strip_preamble_noop_when_already_clean():
    conv = "Doctor: Hello, how are you?\nPatient: Fine."
    cleaned, n = strip_preamble(conv)
    assert cleaned == conv
    assert n == 0


def test_strip_preamble_raises_when_no_turn_marker():
    with pytest.raises(ValueError):
        strip_preamble("just some prose with no dialogue at all")


def test_split_turns_counts_doctor_and_patient_turns():
    conv = "Doctor: Hi.\nPatient: Hi back.\nDoctor: How do you feel?"
    turns = split_turns(conv)
    assert turns == ["Doctor: Hi.", "Patient: Hi back.", "Doctor: How do you feel?"]


def test_build_records_drops_exact_duplicate_notes():
    raw = pl.DataFrame(
        {
            "data": ["same note text", "same note text", "different note text"],
            "conversation": [
                "Doctor: Hi.\nPatient: Hi.",
                "Doctor: Hello.\nPatient: Hello.",
                "Doctor: Hey.\nPatient: Hey.",
            ],
        }
    )
    records, dropped, meta = build_records(raw)
    assert len(records) == 2
    assert len(dropped) == 1


def test_build_records_strips_preamble_and_counts_it():
    raw = pl.DataFrame(
        {
            "data": ["note a", "note b"],
            "conversation": [
                "leaked prompt text\nDoctor: Hi.\nPatient: Hi.",
                "Doctor: Hi.\nPatient: Hi.",
            ],
        }
    )
    records, _, meta = build_records(raw)
    assert meta["preamble_stripped_count"] == 1
    stripped_record = next(r for r in records if r["clinical_note"] == "note a")
    assert stripped_record["preamble_stripped"] is True
    assert stripped_record["conversation"].startswith("Doctor: Hi.")


def test_split_records_no_overlap_and_full_coverage():
    records = [{"note_id": f"note-{i}"} for i in range(100)]
    cfg = {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42}
    splits = split_records(records, cfg)

    train_ids = {r["note_id"] for r in splits["train"]}
    val_ids = {r["note_id"] for r in splits["val"]}
    test_ids = {r["note_id"] for r in splits["test"]}

    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    assert train_ids | val_ids | test_ids == {r["note_id"] for r in records}
    assert len(train_ids) == 80
    assert len(val_ids) == 10
    assert len(test_ids) == 10


def test_split_records_is_deterministic():
    records = [{"note_id": f"note-{i}"} for i in range(50)]
    cfg = {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42}
    splits_a = split_records(records, cfg)
    splits_b = split_records(records, cfg)
    assert {r["note_id"] for r in splits_a["train"]} == {r["note_id"] for r in splits_b["train"]}


def test_load_raw_rejects_unexpected_columns(tmp_path):
    raw = pl.DataFrame({"data": ["x"], "wrong_column": ["y"]})
    bad_csv = tmp_path / "_bad_schema_test.csv"
    raw.write_csv(bad_csv)
    with pytest.raises(ValueError):
        load_raw(bad_csv)


# --- Acceptance tests against the real built dataset ---


def test_no_note_id_overlap_across_real_splits():
    _skip_if_not_built()
    train = pl.read_parquet(PROCESSED_DIR / "train.parquet")["note_id"].to_list()
    val = pl.read_parquet(PROCESSED_DIR / "val.parquet")["note_id"].to_list()
    test = pl.read_parquet(PROCESSED_DIR / "test.parquet")["note_id"].to_list()

    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))


def test_real_conversations_start_with_a_turn_marker():
    _skip_if_not_built()
    df = pl.read_parquet(PROCESSED_DIR / "test.parquet")
    for row in df.iter_rows(named=True):
        assert row["conversation"].startswith("Doctor:") or row["conversation"].startswith("Patient:"), (
            f"{row['note_id']}: conversation does not start with a turn marker"
        )


def test_real_notes_and_conversations_are_non_empty():
    _skip_if_not_built()
    df = pl.read_parquet(PROCESSED_DIR / "train.parquet")
    assert (df["note_len_chars"] > 0).all()
    assert (df["conversation_len_chars"] > 0).all()
    assert (df["num_turns"] > 0).all()
