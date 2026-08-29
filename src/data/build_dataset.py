"""Phase 1 — raw NoteChat CSV -> task-ready parquet (PROJECT_SPEC.md §5 Phase 1).

Input:  data/raw/notechat/our_revised_v2.csv (two columns: `data` = a
        clinical note, `conversation` = a synthetic doctor-patient dialogue
        conditioned on that note — see docs/DECISIONS.md for why this
        variant was picked over NoteChat's other released files).
Output: data/processed/{train,val,test}.parquet, one row per note:
        {clinical_note, conversation} is the task's input/output pair
        (PROJECT_SPEC.md §4). There is no CUAD-style label space here —
        the output is free-form dialogue, not a fixed taxonomy.

Run: python -m src.data.build_dataset
"""

import hashlib
import random
import re
from pathlib import Path

import polars as pl
import yaml

CONFIG_PATH = Path("configs/data.yaml")
REPORT_PATH = Path("docs/data_report.md")

# A generation artifact in NoteChat's source: a small number of conversations
# carry a leftover preamble line (e.g. a leaked keyword list from the
# generation prompt) before the dialogue actually starts. Detected and
# stripped at build time rather than left in the training target — see
# docs/DECISIONS.md for the count this affects.
TURN_START_RE = re.compile(r"^(Doctor|Patient):", re.MULTILINE)
TURN_SPLIT_RE = re.compile(r"(?=^(?:Doctor|Patient):)", re.MULTILINE)


def load_raw(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    expected_cols = {"data", "conversation"}
    if set(df.columns) != expected_cols:
        raise ValueError(f"{path}: expected columns {expected_cols}, got {set(df.columns)}")
    return df


def strip_preamble(conversation: str) -> tuple[str, int]:
    """Drop any text before the first `Doctor:`/`Patient:` turn marker.

    Returns (cleaned_text, chars_stripped). Raises if no turn marker exists
    at all — that's not a preamble artifact, it's a row with no dialogue,
    and should fail loudly rather than be silently kept or silently emptied.
    """
    match = TURN_START_RE.search(conversation)
    if match is None:
        raise ValueError(f"no Doctor:/Patient: turn marker found in conversation: {conversation!r}")
    return conversation[match.start() :], match.start()


def split_turns(conversation: str) -> list[str]:
    parts = TURN_SPLIT_RE.split(conversation)
    return [p.strip() for p in parts if p.strip()]


def build_records(raw: pl.DataFrame) -> tuple[list[dict], list[tuple[str, str]], dict]:
    """Parse raw NoteChat rows into one record per clinical note.

    Drops exact-duplicate notes (by content hash), the same defensive check
    used for CUAD — none exist in the current `our_revised_v2.csv` release
    (verified at build time, see docs/DECISIONS.md), but a future NoteChat
    release could reintroduce them and this should not silently double-count
    a note across splits if it does.
    """
    records = []
    seen_hashes: dict[str, str] = {}
    dropped_dupes: list[tuple[str, str]] = []
    preamble_stripped_count = 0

    for row in raw.iter_rows(named=True):
        note = row["data"]
        conversation_raw = row["conversation"]

        content_hash = hashlib.sha256(note.encode()).hexdigest()
        note_id = f"note-{content_hash[:16]}"
        if content_hash in seen_hashes:
            dropped_dupes.append((note_id, seen_hashes[content_hash]))
            continue
        seen_hashes[content_hash] = note_id

        conversation, chars_stripped = strip_preamble(conversation_raw)
        if chars_stripped > 0:
            preamble_stripped_count += 1

        turns = split_turns(conversation)
        num_doctor_turns = sum(1 for t in turns if t.startswith("Doctor:"))
        num_patient_turns = sum(1 for t in turns if t.startswith("Patient:"))

        records.append(
            {
                "note_id": note_id,
                "clinical_note": note,
                "note_len_chars": len(note),
                "conversation": conversation,
                "conversation_len_chars": len(conversation),
                "num_turns": len(turns),
                "num_doctor_turns": num_doctor_turns,
                "num_patient_turns": num_patient_turns,
                "preamble_stripped": chars_stripped > 0,
            }
        )

    records_meta = {"preamble_stripped_count": preamble_stripped_count}
    return records, dropped_dupes, records_meta


def split_records(records: list[dict], split_cfg: dict) -> dict[str, list[dict]]:
    """Split by note_id (OPEN_DECISIONS #5) — each row is already one unique
    clinical note (verified at build time, no note appears twice), so
    splitting by row is equivalent to splitting by source document, unlike
    CUAD where one contract had many qa rows."""
    note_ids = sorted(r["note_id"] for r in records)
    rng = random.Random(split_cfg["seed"])
    rng.shuffle(note_ids)

    n = len(note_ids)
    n_train = round(n * split_cfg["train"])
    n_val = round(n * split_cfg["val"])
    train_ids = set(note_ids[:n_train])
    val_ids = set(note_ids[n_train : n_train + n_val])
    test_ids = set(note_ids[n_train + n_val :])

    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)
    assert train_ids | val_ids | test_ids == set(note_ids)

    by_id = {r["note_id"]: r for r in records}
    return {
        "train": [by_id[nid] for nid in sorted(train_ids)],
        "val": [by_id[nid] for nid in sorted(val_ids)],
        "test": [by_id[nid] for nid in sorted(test_ids)],
    }


def write_report(
    all_records: list[dict],
    splits: dict[str, list[dict]],
    dropped_dupes: list[tuple[str, str]],
    preamble_stripped_count: int,
    source_file: str,
) -> None:
    n = len(all_records)

    def stats(values: list[int]) -> str:
        return f"mean {round(sum(values) / len(values), 1)}, min {min(values)}, max {max(values)}"

    lines = [
        "# Data report — NoteChat (Phase 1)",
        "",
        "Generated by `src/data/build_dataset.py`. Do not hand-edit — rerun the",
        "script to regenerate after any pipeline change.",
        "",
        "## Coverage",
        "",
        f"- Source file: `{source_file}` (see `docs/DECISIONS.md` for why this",
        "  variant was picked over NoteChat's other released files).",
        f"- Raw rows: {n + len(dropped_dupes)}",
        f"- Exact-duplicate notes dropped (by content hash): {len(dropped_dupes)}",
        f"- Notes in the final dataset: {n}",
        "",
        "## Splits (by `note_id`, seed 42 — see `configs/data.yaml`)",
        "",
        "| Split | Notes | % |",
        "|---|---|---|",
    ]
    for name, recs in splits.items():
        lines.append(f"| {name} | {len(recs)} | {round(100 * len(recs) / n, 1)}% |")

    lines += [
        "",
        "## Data-quality notes",
        "",
        f"- Conversations with a leading preamble before the first "
        f"`Doctor:`/`Patient:` turn marker (stripped at build time): "
        f"{preamble_stripped_count} ({round(100 * preamble_stripped_count / n, 2)}%). "
        "This is a generation artifact in the source data (e.g. a leaked "
        "keyword list from the prompt), not a labeling error — see "
        "`docs/DECISIONS.md`.",
        "- **Ground-truth caveat:** the `conversation` column is itself an "
        "LLM-generated synthetic dialogue (NoteChat's multi-agent pipeline "
        "output), not a human-authored or human-verified reference. Treat "
        "reference-based metrics (ROUGE/BLEU/BERTScore against this column) "
        "as *similarity to one synthetic exemplar*, not as ground truth in "
        "the sense CUAD's expert annotations were. See `PROJECT_SPEC.md` "
        "§4.2 and Phase 4.",
        "",
        "## Clinical note length (chars)",
        "",
        f"- {stats([r['note_len_chars'] for r in all_records])}",
        "",
        "## Conversation length (chars)",
        "",
        f"- {stats([r['conversation_len_chars'] for r in all_records])}",
        "",
        "## Turns per conversation",
        "",
        f"- Total turns: {stats([r['num_turns'] for r in all_records])}",
        f"- Doctor turns: {stats([r['num_doctor_turns'] for r in all_records])}",
        f"- Patient turns: {stats([r['num_patient_turns'] for r in all_records])}",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def to_parquet(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(records).write_parquet(path)


def main() -> None:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    raw_path = Path(cfg["data_root"]) / cfg["source_file"]
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. NoteChat's CSVs are gitignored (see .gitignore) "
            f"and expected under data/raw/notechat/ — restore from the original "
            f"akemiH/NoteChat HuggingFace dataset download if missing."
        )

    raw = load_raw(raw_path)
    records, dropped_dupes, meta = build_records(raw)
    splits = split_records(records, cfg["split"])

    out_dir = Path(cfg["output_dir"])
    for name, recs in splits.items():
        to_parquet(recs, out_dir / f"{name}.parquet")

    write_report(records, splits, dropped_dupes, meta["preamble_stripped_count"], cfg["source_file"])

    print(f"Wrote {len(records)} notes across {len(splits)} splits to {out_dir}/")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
