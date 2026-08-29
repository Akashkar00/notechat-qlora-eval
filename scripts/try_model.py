"""Manual probe of a trained arm — hand it a clinical note, read the dialogue.

Every other entrypoint in this repo reports aggregates. This one exists for
the thing aggregates cannot do: looking at a single generation and judging it
yourself. The Phase 5 fabrication finding (`docs/DECISIONS.md`) came from
exactly this kind of manual read, not from a metric.

It reuses `src/inference/local_hf.py` for prompting and
`src/eval/faithfulness.py` for scoring, so what you see here is generated and
scored by the same code paths `run_eval.py` uses — a manual spot-check that
disagrees with the harness means one of the two is wrong, which is the point
of sharing the code rather than re-implementing it here.

Examples
--------
    # Fine-tuned model on a random held-out test note, next to its reference
    python scripts/try_model.py --from-test random

    # The same note through the base model and the fine-tune, side by side
    python scripts/try_model.py --from-test 7 --compare

    # Your own note
    python scripts/try_model.py --note-file mynote.txt
    python scripts/try_model.py --note "A 64-year-old man presented with..."

    # Paste a note on stdin (Ctrl+Z then Enter on Windows, Ctrl+D on Unix)
    python scripts/try_model.py

    # Keep the model loaded and ask repeatedly (~25s of load, then seconds each)
    python scripts/try_model.py --repl

Decoding is greedy by default, matching the eval harness, so what you see is
what `run_eval.py` scored. Pass --sample for a qualitative look at output
diversity — the one thing greedy decoding hides.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402

from src.eval.faithfulness import extract_numbers, numeric_faithfulness  # noqa: E402
from src.eval.metrics import rouge, turn_format_validity  # noqa: E402
from src.inference.local_hf import generate, load_model  # noqa: E402

DEFAULT_ADAPTER = "artifacts/adapters/unsloth__Qwen2.5-3B-Instruct-bnb-4bit/final_adapter"
DEFAULT_BASE = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
TEST_PARQUET = Path("data/processed/test.parquet")
RULE = "=" * 78


def load_test_note(selector: str, seed: int) -> dict:
    """Pull one record from the held-out test split by index, note_id, or at random."""
    if not TEST_PARQUET.exists():
        raise SystemExit(f"{TEST_PARQUET} not found — run `python -m src.data.build_dataset` first.")
    df = pl.read_parquet(TEST_PARQUET)

    if selector == "random":
        idx = random.Random(seed).randrange(len(df))
    elif selector.startswith("note-"):
        matches = df.filter(pl.col("note_id") == selector)
        if len(matches) == 0:
            raise SystemExit(f"note_id {selector!r} is not in the test split.")
        return matches.row(0, named=True)
    else:
        try:
            idx = int(selector)
        except ValueError:
            raise SystemExit(f"--from-test wants an index, a note_id, or 'random'; got {selector!r}") from None
        if not 0 <= idx < len(df):
            raise SystemExit(f"--from-test index {idx} is out of range (test split holds {len(df)}).")
    return df.row(idx, named=True)


def report(label: str, note: str, generated: str, reference: str | None) -> None:
    """Print one generation alongside the same checks run_eval.py would apply."""
    faith = numeric_faithfulness(generated, note)
    fmt = turn_format_validity(generated)

    print(f"\n{RULE}\n{label}\n{RULE}")
    print(generated.strip())

    print(f"\n--- scored against the NOTE ({faith['n_note_numbers']} numbers in it) ---")
    print(f"  numeric grounding recall : {faith['numeric_grounding_recall']:.3f}   (how much of the note survived)")
    print(
        f"  fabricated number rate   : {faith['fabricated_number_rate']:.3f}   "
        f"(of the {faith['n_generated_numbers']} numbers it stated)"
    )
    print(
        f"  turns                    : {fmt['num_turns']} "
        f"(doctor={fmt['has_doctor_turn']}, patient={fmt['has_patient_turn']})"
    )

    if reference is not None:
        r = rouge(generated, reference)
        print(f"  ROUGE-1 / ROUGE-L vs ref : {r['rouge1']:.3f} / {r['rougeL']:.3f}")

    invented = sorted(extract_numbers(generated) - extract_numbers(note), key=float)
    if invented:
        print(f"  numbers NOT in the note  : {', '.join(invented)}")
        print("    ^ read these by hand. Some are harmless (a turn count, '2 weeks' the")
        print("      patient volunteers); some are invented clinical values. The metric")
        print("      cannot tell the difference — you can. That gap is why this exists.")


def build_arms(args: argparse.Namespace) -> list[tuple[str, str | None]]:
    """(label, adapter_path) per arm to run, in load order."""
    if args.compare:
        return [("BASE (zero-shot, arm 1)", None), ("FINE-TUNED (QLoRA, arm 3)", args.adapter)]
    if args.base:
        return [("BASE (zero-shot, arm 1)", None)]
    return [("FINE-TUNED (QLoRA, arm 3)", args.adapter)]


def read_note_from_repl() -> str | None:
    """Read a multi-line note; blank line ends it. None means the user is done."""
    lines: list[str] = []
    print("\nnote> ", end="", flush=True)
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            return None
        if line == "" and lines:
            return "\n".join(lines)
        if line:
            lines.append(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a doctor-patient dialogue from a clinical note and inspect it by hand.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--note", help="Clinical note text, inline.")
    source.add_argument("--note-file", help="Path to a file holding the clinical note.")
    source.add_argument("--from-test", metavar="SEL", help="Index, note_id, or 'random' from the held-out test split.")

    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="LoRA adapter dir (default: the committed one).")
    parser.add_argument("--base", action="store_true", help="Use the frozen base model instead of the adapter.")
    parser.add_argument("--compare", action="store_true", help="Run base AND fine-tuned on the same note.")
    parser.add_argument("--model-name", default=DEFAULT_BASE, help="Base checkpoint id.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sample", action="store_true", help="Stochastic decoding (default greedy, as in eval).")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repl", action="store_true", help="Keep the model loaded and prompt for notes repeatedly.")
    args = parser.parse_args()

    note: str | None = None
    reference: str | None = None

    if args.from_test:
        row = load_test_note(args.from_test, args.seed)
        note, reference = row["clinical_note"], row["conversation"]
        print(f"Test record: {row['note_id']}")
    elif args.note_file:
        note = Path(args.note_file).read_text(encoding="utf-8")
    elif args.note:
        note = args.note
    elif not args.repl:
        print("Paste the clinical note, then Ctrl+Z + Enter (Windows) / Ctrl+D (Unix):", file=sys.stderr)
        note = sys.stdin.read()

    if note is not None and not note.strip():
        raise SystemExit("Empty clinical note — nothing to generate from.")

    if note is not None:
        shown = note.strip()
        print(f"\n{RULE}\nCLINICAL NOTE ({len(note)} chars)\n{RULE}")
        print(
            shown[:1500] + ("\n[...truncated for display only; the model sees all of it]" if len(shown) > 1500 else "")
        )

    for label, adapter in build_arms(args):
        if adapter is not None and not Path(adapter).exists():
            raise SystemExit(f"Adapter not found at {adapter} — train first, or pass --base.")

        print(f"\nLoading {label} ...", file=sys.stderr)
        model, tokenizer = load_model(
            model_name=args.model_name,
            max_seq_len=args.max_seq_len,
            load_in_4bit=True,
            seed=args.seed,
            adapter_path=adapter,
        )

        def run_one(text: str, ref: str | None, label: str = label, model=model, tokenizer=tokenizer) -> None:
            gen = generate(
                model,
                tokenizer,
                {"clinical_note": text},
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.sample,
            )
            report(label, text, gen, ref)

        if note is not None:
            run_one(note, reference)

        if args.repl:
            print(f"\n{RULE}\nREPL — paste a note, then a blank line. Ctrl+C or Ctrl+D to quit.\n{RULE}")
            while True:
                text = read_note_from_repl()
                if text is None:
                    print("\nbye")
                    return
                if text.strip():
                    # No reference for a note you supplied — there isn't one.
                    run_one(text, None)

    if reference is not None:
        print(f"\n{RULE}\nREFERENCE — NoteChat's own dialogue for this note\n{RULE}")
        print(reference.strip())
        print(
            "\nNote this is itself LLM-generated (PROJECT_SPEC.md §4.2), not a real\n"
            "transcript. It is what ROUGE/BERTScore score against — so judge the\n"
            "generations above on the note, not on how closely they match this."
        )


if __name__ == "__main__":
    main()
