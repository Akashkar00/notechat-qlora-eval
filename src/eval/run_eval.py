"""Phase 2 eval harness entrypoint (PROJECT_SPEC.md §5 Phase 2).

Runs one model "arm" (zero-shot base model, or a fine-tuned adapter) over a
sample of the held-out test split, computes reference-based generation
metrics + a structural format check + latency/VRAM, and writes a complete
results JSON. This is the acceptance test for Phase 2 and doubles as Phase
5/6's "eval harness produces a full result set on the test split" /
per-arm baseline runs.

Usage:
    python -m src.eval.run_eval --arm zero-shot
    python -m src.eval.run_eval --arm finetuned \
        --adapter artifacts/adapters/unsloth__Qwen2.5-3B-Instruct-bnb-4bit/final_adapter

Caveat carried from PROJECT_SPEC.md §4.2: the `conversation` reference
column is itself LLM-generated, not ground truth — ROUGE/BERTScore here
measure similarity to one synthetic exemplar, not correctness.
"""

import argparse
import json
import time
from pathlib import Path

import polars as pl
import torch
import yaml

from src.eval.metrics import bertscore, bootstrap_ci, rouge, turn_format_validity
from src.inference.local_hf import generate, load_model

TRAIN_CONFIG_PATH = Path("configs/train.yaml")
EVAL_CONFIG_PATH = Path("configs/eval.yaml")
TEST_PARQUET = Path("data/processed/test.parquet")
RESULTS_DIR = Path("artifacts/eval")

BOOTSTRAPPED_METRICS = ["rouge1", "rouge2", "rougeL", "bertscore_f1"]


def sample_records(df: pl.DataFrame, n: int | None, seed: int) -> pl.DataFrame:
    if n is None or n >= len(df):
        return df
    return df.sample(n=n, seed=seed)


def run(arm: str, model_name: str, adapter_path: str | None, n_records: int | None, seed: int) -> dict:
    train_cfg = yaml.safe_load(TRAIN_CONFIG_PATH.read_text())
    eval_cfg = yaml.safe_load(EVAL_CONFIG_PATH.read_text())

    if not TEST_PARQUET.exists():
        raise FileNotFoundError(f"{TEST_PARQUET} not found — run `python -m src.data.build_dataset` first.")
    df = sample_records(pl.read_parquet(TEST_PARQUET), n_records, seed)

    model, tokenizer = load_model(
        model_name=model_name,
        max_seq_len=train_cfg["max_seq_len"],
        load_in_4bit=train_cfg["quantization"]["load_in_4bit"],
        seed=train_cfg["seed"],
        adapter_path=adapter_path,
    )

    torch.cuda.reset_peak_memory_stats()
    records = []
    total_gen_tokens = 0
    t0 = time.perf_counter()
    for row in df.iter_rows(named=True):
        row_t0 = time.perf_counter()
        gen = generate(model, tokenizer, row)
        row_elapsed = time.perf_counter() - row_t0

        n_tokens = len(tokenizer(gen, add_special_tokens=False)["input_ids"])
        total_gen_tokens += n_tokens

        r = rouge(gen, row["conversation"])
        fmt = turn_format_validity(gen)
        records.append(
            {
                "note_id": row["note_id"],
                "generated": gen,
                "reference": row["conversation"],
                "rouge1": r["rouge1"],
                "rouge2": r["rouge2"],
                "rougeL": r["rougeL"],
                "gen_tokens": n_tokens,
                "latency_s": row_elapsed,
                **fmt,
            }
        )
    total_elapsed = time.perf_counter() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    bert = bertscore([r["generated"] for r in records], [r["reference"] for r in records])
    for rec, f1 in zip(records, bert["f1"]):
        rec["bertscore_f1"] = f1

    bs_cfg = eval_cfg["bootstrap"]
    aggregate = {
        metric: bootstrap_ci(
            [r[metric] for r in records],
            n_resamples=bs_cfg["n_resamples"],
            confidence_level=bs_cfg["confidence_level"],
        )
        for metric in BOOTSTRAPPED_METRICS
    }
    n = len(records)
    aggregate["turn_format_valid_rate"] = (
        sum(r["starts_with_turn_marker"] and r["has_doctor_turn"] and r["has_patient_turn"] for r in records) / n
    )
    aggregate["tokens_per_second"] = total_gen_tokens / total_elapsed
    aggregate["peak_vram_gb"] = peak_vram_gb
    aggregate["wall_clock_s"] = total_elapsed

    result = {
        "arm": arm,
        "model_name": model_name,
        "adapter_path": adapter_path,
        "n_records": n,
        "aggregate": aggregate,
        "records": records,
    }

    out_path = RESULTS_DIR / arm / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, help="Name for this run, e.g. 'zero-shot' or 'finetuned'")
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    parser.add_argument("--adapter", default=None, help="Path to a saved LoRA adapter dir; omit for zero-shot")
    parser.add_argument("--n-records", type=int, default=None, help="Defaults to configs/eval.yaml's val_records")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_cfg = yaml.safe_load(EVAL_CONFIG_PATH.read_text())
    n_records = args.n_records if args.n_records is not None else eval_cfg["val_records"]

    run(args.arm, args.model_name, args.adapter, n_records, args.seed)


if __name__ == "__main__":
    main()
