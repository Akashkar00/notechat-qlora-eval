"""Pre-commit hook: block data-restricted paths and sensitive-data-shaped content.

Exits non-zero (blocking the commit) if a staged file sits under a
data-restricted path or has a data-file extension. This is a first line of
defense, not a substitute for keeping data/ out of git via .gitignore.

SUSPICIOUS_PATTERNS covers two things:

1. **The raw NoteChat schema.** Its CSV header (`data,conversation`) pasted
   into any committed file means a chunk of the raw corpus came with it.
   The `.csv` extension rule already blocks the file itself; this catches
   the same content smuggled into a `.py`, `.md`, or `.json`.

2. **PHI-shaped identifiers.** This harness is built to the rule that the
   underlying data could never leave the machine (PROJECT_SPEC.md §1.1).
   NoteChat is public research data, but the whole point of the pattern list
   is that it keeps working when the same harness is pointed at a real
   clinical or company corpus — so it screens for the identifier shapes that
   would actually matter then: MRN/SSN/NHS numbers, dates of birth, phone
   numbers and emails. This is the same role clinical-coding-eval's
   check_phi_leak.py plays for MIMIC column names (docs/DECISIONS.md).

Deliberately *not* pattern-matched: `Doctor:`/`Patient:` turn markers. They
appear legitimately throughout the tests, docstrings and docs (25 times in
tests/test_data.py alone), so flagging them would make the hook noise that
gets bypassed rather than a check anyone trusts.

Known and accepted: `artifacts/eval/*/results.json` embeds NoteChat
reference dialogues and model generations. That is deliberate — they are the
evidence this project produces, and NoteChat is public (PROJECT_SPEC.md §7a
item 3). ALLOWLISTED_PATHS keeps the hook from fighting that decision on
every commit; if this harness is ever repointed at non-public data, that
allowlist is the first thing to remove.
"""

import re
import sys
from pathlib import Path

# Anchored to the repo root, for the same reason .gitignore's rules are:
# testing `"data" in path.parts` matches a `data` directory at ANY depth, so
# it blocked every commit touching `src/data/` — the entire Phase 1 pipeline.
# That is the identical bug .gitignore carried (see its header comment); this
# hook silently had it too, and would have rejected those commits had
# `pre-commit install` ever been run.
BLOCKED_TOP_LEVEL_DIRS = {"data", "models", "outputs"}
# Gold annotations, wherever they live (annotation/gold*/ is the expected spot).
GOLD_DIR_PREFIX = "gold"
BLOCKED_EXTENSIONS = {".parquet", ".csv", ".jsonl"}

# Committed on purpose: these hold NoteChat dialogues because they ARE the
# project's evidence (see module docstring). Path checks still apply to them;
# only the content patterns are skipped.
ALLOWLISTED_PATHS = {Path("artifacts/eval")}

SUSPICIOUS_PATTERNS: list[re.Pattern] = [
    # Raw NoteChat CSV header — the corpus's own schema, pasted anywhere.
    re.compile(r"^\s*data\s*,\s*conversation\s*$", re.MULTILINE),
    # US SSN.
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Medical record number, written out with its label.
    re.compile(r"\b(?:MRN|medical\s+record\s+(?:number|no\.?))\s*[:#]?\s*\d{5,}", re.IGNORECASE),
    # UK NHS number (10 digits, conventionally grouped 3-3-4).
    re.compile(r"\b\d{3}[ -]\d{3}[ -]\d{4}\b"),
    # A labelled date of birth — the label is what makes this PHI rather than
    # just any date, and is what keeps it off ordinary dates in the docs.
    re.compile(r"\b(?:DOB|date\s+of\s+birth)\s*[:#]?\s*\d", re.IGNORECASE),
    # Email address.
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
]


def check_file(path: Path) -> list[str]:
    reasons = []
    parts = path.parts
    if parts and parts[0] in BLOCKED_TOP_LEVEL_DIRS:
        reasons.append(f"{path} is under the data-restricted top-level directory '{parts[0]}/'")
    if any(part.startswith(GOLD_DIR_PREFIX) for part in parts[:-1]):
        reasons.append(f"{path} is under a gold-annotation directory")
    if path.suffix.lower() in BLOCKED_EXTENSIONS:
        reasons.append(f"{path} has a data-file extension ({path.suffix})")

    if any(allowed in path.parents for allowed in ALLOWLISTED_PATHS):
        return reasons

    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return reasons

    for pattern in SUSPICIOUS_PATTERNS:
        if pattern.search(text):
            reasons.append(f"{path} matches a suspected sensitive-data pattern: {pattern.pattern}")
    return reasons


def main(argv: list[str]) -> int:
    blocked = False
    for raw in argv:
        path = Path(raw)
        if not path.is_file():
            continue
        for reason in check_file(path):
            print(f"BLOCKED: {reason}")
            blocked = True
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
