"""Tests for the pre-commit data-leak hook (scripts/check_data_leak.py).

The hook is only worth having if it blocks what it claims to block and stays
quiet on the repo's own files — a hook that fires on legitimate commits gets
bypassed, at which point it protects nothing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_data_leak import check_file  # noqa: E402


def _write(tmp_path: Path, relative: str, content: str) -> Path:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_blocks_top_level_data_directory(tmp_path):
    _write(tmp_path, "data/train.csv", "note_id,text\n1,hello\n")
    assert check_file(Path("data/train.csv"))


def test_does_not_block_src_data_package(tmp_path):
    """The regression this hook shipped with: an unanchored `data` part
    matched `src/data/`, so the hook would have rejected every commit
    touching the Phase 1 pipeline — the same bug .gitignore had."""
    path = _write(tmp_path, "src/data/build_dataset.py", "import polars as pl\n")
    assert check_file(path) == []


def test_blocks_gold_annotation_directory():
    assert check_file(Path("annotation/gold_v1/labels.txt"))


def test_blocks_data_file_extensions():
    for name in ("x.parquet", "x.csv", "x.jsonl"):
        assert check_file(Path(name)), name


def test_blocks_pasted_notechat_csv_header(tmp_path):
    path = _write(tmp_path, "notes.md", "data,conversation\nsome note,some dialogue\n")
    assert any("sensitive-data pattern" in r for r in check_file(path))


def test_blocks_phi_shaped_identifiers(tmp_path):
    cases = {
        "ssn.md": "Patient SSN 123-45-6789 seen today\n",
        "mrn.md": "MRN: 4457821 admitted overnight\n",
        "dob.md": "DOB: 1974-02-11\n",
        "email.md": "contact jane.doe@hospital.org\n",
        "nhs.md": "NHS 943 476 5919\n",
    }
    for name, content in cases.items():
        path = _write(tmp_path, name, content)
        assert any("sensitive-data pattern" in r for r in check_file(path)), name


def test_allows_ordinary_clinical_numbers(tmp_path):
    """Ages, vitals and plain dates are the substance of every clinical note
    in this corpus. Flagging them would make the hook unusable."""
    path = _write(tmp_path, "notes.md", "The patient was 64 years old, BP 135/85, seen on 2015-03-12.\n")
    assert check_file(path) == []


def test_allows_dialogue_turn_markers(tmp_path):
    """Doctor:/Patient: appear all over the tests and docs by design."""
    path = _write(tmp_path, "notes.md", "Doctor: Hello.\nPatient: Hi.\nDoctor: How are you feeling?\n")
    assert check_file(path) == []


def test_committed_eval_artifacts_are_allowlisted(tmp_path):
    """artifacts/eval/*/results.json embeds NoteChat dialogues on purpose —
    they are the evidence the project produces."""
    _write(tmp_path, "results.json", '{"reference": "Doctor: data,conversation"}')
    assert check_file(Path("artifacts/eval/finetuned/results.json")) == []
