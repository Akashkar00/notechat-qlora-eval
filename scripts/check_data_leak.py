"""Pre-commit hook: block data-restricted paths and sensitive-data-shaped content.

Exits non-zero (blocking the commit) if a staged file sits under a
data-restricted path or has a data-file extension. This is a first line of
defense, not a substitute for keeping data/ out of git via .gitignore.

SUSPICIOUS_PATTERNS starts empty because the real data schema is unknown
until PROJECT_SPEC.md's OPEN_DECISIONS is filled in. Once the data source
is known, add regex patterns here for its identifying column
headers/markers, the same way clinical-coding-eval's check_phi_leak.py
does for MIMIC column names — see docs/DECISIONS.md for that precedent.
"""

import re
import sys
from pathlib import Path

BLOCKED_PATH_PARTS = {"data", "gold"}
BLOCKED_EXTENSIONS = {".parquet", ".csv", ".jsonl"}

SUSPICIOUS_PATTERNS: list[re.Pattern] = [
    # Add data-source-specific patterns once OPEN_DECISIONS #2 is answered.
]


def check_file(path: Path) -> list[str]:
    reasons = []
    if any(part in BLOCKED_PATH_PARTS for part in path.parts):
        reasons.append(f"{path} is under a data-restricted path")
    if path.suffix.lower() in BLOCKED_EXTENSIONS:
        reasons.append(f"{path} has a data-file extension ({path.suffix})")

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
