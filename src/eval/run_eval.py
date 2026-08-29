"""Phase 2 eval harness entrypoint (PROJECT_SPEC.md §5 Phase 2).

Runs one model "arm" over a sample of the held-out test split, computes
reference-based generation metrics + a structural format check + a
faithfulness proxy + latency/VRAM, and writes a complete results JSON. This
is the acceptance test for Phase 2 and doubles as Phase 5/6's per-arm
baseline runs.

Usage:
    # arm 1 — zero-shot small model
    python -m src.eval.run_eval --arm zero-shot

    # arm 2 — zero-shot large baseline
    python -m src.eval.run_eval --arm zero-shot-14b \
        --model-name unsloth/Qwen2.5-14B-Instruct-bnb-4bit --max-seq-len 2048

    # arm 3 — QLoRA fine-tuned small model
    python -m src.eval.run_eval --arm finetuned \
        --adapter artifacts/adapters/unsloth__Qwen2.5-3B-Instruct-bnb-4bit/final_adapter

    # arm 4 — classic non-LLM baseline (no GPU generation)
    python -m src.eval.run_eval --arm classic-tfidf --baseline tfidf

Decoding is **greedy by default** so a rerun of any command above reproduces
its numbers exactly (PROJECT_SPEC.md §1.2). Pass --sample to opt into
stochastic decoding; the chosen config is recorded in the results JSON so no
result is ambiguous about how it was produced.

Caveat carried from PROJECT_SPEC.md §4.2: the `conversation` reference
column is itself LLM-generated, not ground truth — ROUGE/BERTScore here
measure similarity to one synthetic exemplar, not correctness. The
faithfulness metrics are scored against the clinical note instead, which is
real.
"""

import argparse
import json
import time
from pathlib import Path

import polars as pl
import torch
import yaml

from src.eval.faithfulness import numeric_faithfulness
from src.eval.metrics import bertscore, bootstrap_ci, rouge, turn_format_validity
from src.inference.local_hf import generate, load_model

TRAIN_CONFIG_PATH = Path("configs/train.yaml")
EVAL_CONFIG_PATH = Path("configs/eval.yaml")
TRAIN_PARQUET = Path("data/processed/train.parquet")
TEST_PARQUET = Path("data/processed/test.parquet")
RESULTS_DIR = Path("artifacts/eval")

BOOTSTRAPPED_METRICS = [
    "rouge1",
    "rouge2",
    "rougeL",
    "bertscore_f1",
    "numeric_grounding_recall",
    "numeric_precision",
    "fabricated_number_rate",
]


def sample_records(df: pl.DataFrame, n: int | None, seed: int) -> pl.DataFrame:
    if n is None or n >= len(df):
        return df
    return df.sample(n=n, seed=seed)


def _score_record(row: dict, gen: str, extra: dict) -> dict:
    """Score one generation against both its reference dialogue and its
    source note. Shared by every arm so a classic baseline and an LLM arm
    are never scored by slightly different code."""
    r = rouge(gen, row["conversation"])
    fmt = turn_format_validity(gen)
    faith = numeric_faithfulness(gen, row["clinical_note"])
    return {
        "note_id": row["note_id"],
        "generated": gen,
        "reference": row["conversation"],
        "rouge1": r["rouge1"],
        "rouge2": r["rouge2"],
        "rougeL": r["rougeL"],
        **fmt,
        **faith,
        **extra,
    }


def run(
    arm: str,
    model_name: str,
    adapter_path: str | None,
    n_records: int | None,
    seed: int,
    max_seq_len: int | None = None,
    baseline: str | None = None,
    do_sample: bool = False,
) -> dict:
    train_cfg = yaml.safe_load(TRAIN_CONFIG_PATH.read_text())
    eval_cfg = yaml.safe_load(EVAL_CONFIG_PATH.read_text())

    if not TEST_PARQUET.exists():
        raise FileNotFoundError(f"{TEST_PARQUET} not found — run `python -m src.data.build_dataset` first.")
    df = sample_records(pl.read_parquet(TEST_PARQUET), n_records, seed)

    # Arm 4 is advertised as needing no GPU, so the CUDA bookkeeping has to be
    # optional rather than unconditional — torch.cuda.reset_peak_memory_stats()
    # raises on a CPU-only machine, which would make the "no model, no GPU"
    # baseline the one arm that still demands CUDA to run.
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
    records = []
    total_gen_tokens = 0

    if baseline == "tfidf":
        # Arm 4 — no model, no GPU. Fit on train only: retrieving from the
        # test split itself would leak the answer, and retrieving from a
        # pool containing the query note would make this arm trivially and
        # meaninglessly perfect.
        from src.baselines.classic_baseline import TfidfRetrievalBaseline

        if not TRAIN_PARQUET.exists():
            raise FileNotFoundError(f"{TRAIN_PARQUET} not found — run `python -m src.data.build_dataset` first.")
        retriever = TfidfRetrievalBaseline().fit(pl.read_parquet(TRAIN_PARQUET))

        t0 = time.perf_counter()
        for row in df.iter_rows(named=True):
            row_t0 = time.perf_counter()
            pred = retriever.predict_one(row["clinical_note"])
            row_elapsed = time.perf_counter() - row_t0
            records.append(
                _score_record(
                    row,
                    pred["generated"],
                    {
                        "latency_s": row_elapsed,
                        "retrieved_note_id": pred["retrieved_note_id"],
                        "retrieval_similarity": pred["retrieval_similarity"],
                    },
                )
            )
        total_elapsed = time.perf_counter() - t0
    else:
        model, tokenizer = load_model(
            model_name=model_name,
            # configs/train.yaml's max_seq_len (4096) was sized for the 3B
            # fine-tune's training context; a larger zero-shot baseline model
            # needs its own override so its bigger KV cache doesn't need the
            # same headroom this config assumed (PROJECT_SPEC.md §5 Phase 6 arm 2).
            max_seq_len=max_seq_len if max_seq_len is not None else train_cfg["max_seq_len"],
            load_in_4bit=train_cfg["quantization"]["load_in_4bit"],
            seed=train_cfg["seed"],
            adapter_path=adapter_path,
        )

        t0 = time.perf_counter()
        for row in df.iter_rows(named=True):
            row_t0 = time.perf_counter()
            gen, n_tokens = generate(model, tokenizer, row, do_sample=do_sample, return_n_tokens=True)
            row_elapsed = time.perf_counter() - row_t0

            total_gen_tokens += n_tokens
            records.append(_score_record(row, gen, {"latency_s": row_elapsed, "gen_tokens": n_tokens}))
        total_elapsed = time.perf_counter() - t0

    generation_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if cuda_available else None

    # BERTScore loads its own roberta-large scorer onto the GPU, so measure
    # generation VRAM before this point — otherwise every arm's "peak VRAM"
    # would be contaminated by the scorer and stop reflecting serving cost.
    bert = bertscore([r["generated"] for r in records], [r["reference"] for r in records])
    # strict=True: one BERTScore per record, by construction. A length
    # mismatch would silently drop records from the aggregate rather than
    # fail, which is exactly the index-alignment bug compare.py exists to
    # prevent downstream.
    for rec, f1 in zip(records, bert["f1"], strict=True):
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
    aggregate["tokens_per_second"] = total_gen_tokens / total_elapsed if total_gen_tokens else None
    aggregate["records_per_second"] = n / total_elapsed
    aggregate["peak_vram_gb"] = generation_vram_gb
    aggregate["wall_clock_s"] = total_elapsed

    result = {
        "arm": arm,
        "model_name": None if baseline else model_name,
        "adapter_path": adapter_path,
        "baseline": baseline,
        "n_records": n,
        # Recorded so a results file is never ambiguous about how it was
        # produced — greedy vs. sampled changes what the CIs mean.
        "decoding": {"do_sample": do_sample, "greedy": not do_sample} if not baseline else {"deterministic": True},
        "sample_seed": seed,
        "aggregate": aggregate,
        "records": records,
    }

    out_path = RESULTS_DIR / arm / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, help="Name for this run, e.g. 'zero-shot' or 'finetuned'")
    parser.add_argument("--model-name", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    parser.add_argument("--adapter", default=None, help="Path to a saved LoRA adapter dir; omit for zero-shot")
    parser.add_argument("--n-records", type=int, default=None, help="Defaults to configs/eval.yaml's val_records")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-seq-len", type=int, default=None, help="Overrides configs/train.yaml's max_seq_len for this run"
    )
    parser.add_argument(
        "--baseline",
        choices=["tfidf"],
        default=None,
        help="Run a non-LLM baseline instead of a model (Phase 6 arm 4)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use stochastic decoding instead of greedy. Off by default so results are reproducible.",
    )
    args = parser.parse_args()

    eval_cfg = yaml.safe_load(EVAL_CONFIG_PATH.read_text())
    n_records = args.n_records if args.n_records is not None else eval_cfg["val_records"]

    run(
        args.arm,
        args.model_name,
        args.adapter,
        n_records,
        args.seed,
        args.max_seq_len,
        baseline=args.baseline,
        do_sample=args.sample,
    )


if __name__ == "__main__":
    main()
