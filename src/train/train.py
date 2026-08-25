"""Phase 5 — QLoRA fine-tuning (PROJECT_SPEC.md §5 Phase 5).

Task (PROJECT_SPEC.md §4): given a contract's full text, generate every
{clause_type, evidence_span} pair for clause types present, where
evidence_span is a verbatim substring of the input (checked by
src/eval/grounding.py at eval time). Multi-span clause types (see
docs/data_report.md) get one output entry per span, not one per clause
type — collapsing to one span per type would silently drop ground truth
the model is supposed to learn to produce.

Model + hyperparameters: configs/train.yaml. model_name was left unset
there deliberately (PROJECT_SPEC.md §5 — "confirm at build time, do not
assume"); it is passed on the command line instead of hardcoded here so
the choice stays visible in the run command, not buried in a config diff.

Run: python -m src.train.train --model-name Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
from pathlib import Path

import polars as pl
import yaml

CONFIG_PATH = Path("configs/train.yaml")
TRAIN_PARQUET = Path("data/processed/train.parquet")
VAL_PARQUET = Path("data/processed/val.parquet")
ADAPTER_OUT = Path("artifacts/adapters")

SYSTEM_PROMPT = (
    "You are a contract review assistant. Given the full text of a commercial "
    "contract and a fixed list of clause types, identify every clause type that "
    "is present and, for each, quote the exact verbatim text span from the "
    "contract that supports it. A clause type may appear more than once if it "
    "has multiple supporting spans. Output ONLY a JSON array of objects with "
    'keys "clause_type" and "evidence_span" — no other text. If no listed '
    "clause type is present, output an empty JSON array: []."
)


def label_space_from(df: pl.DataFrame) -> list[str]:
    """Derive the closed 41-type taxonomy from the data itself (OPEN_DECISIONS
    #1 — fixed list, not empirically derived) rather than hardcoding it, so a
    schema change in build_dataset.py can't silently drift out of sync."""
    row = df.row(0, named=True)
    space = sorted(set(row["clause_types_present"]) | set(row["clause_types_absent"]))
    assert len(space) == 41, f"expected CUAD's 41 clause types, got {len(space)}"
    return space


def build_target(row: dict) -> str:
    pairs = [
        {"clause_type": clause["clause_type"], "evidence_span": span}
        for clause in row["clause_labels"]
        for span in clause["evidence_spans"]
    ]
    return json.dumps(pairs, ensure_ascii=False)


def build_user_prompt(row: dict, label_space: list[str]) -> str:
    types_list = "\n".join(f"- {t}" for t in label_space)
    return f"Clause types to check for:\n{types_list}\n\nContract text:\n{row['context']}"


def to_chat_text(row: dict, label_space: list[str], tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(row, label_space)},
        {"role": "assistant", "content": build_target(row)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def load_split(path: Path, label_space: list[str], tokenizer):
    from datasets import Dataset

    df = pl.read_parquet(path)
    texts = [to_chat_text(row, label_space, tokenizer) for row in df.iter_rows(named=True)]
    return Dataset.from_dict({"text": texts})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        required=True,
        help="HF checkpoint id, e.g. Qwen/Qwen2.5-3B-Instruct. Not defaulted — "
        "PROJECT_SPEC.md §5 requires confirming this at build time, not assuming it.",
    )
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    if not TRAIN_PARQUET.exists():
        raise FileNotFoundError(f"{TRAIN_PARQUET} not found — run `python -m src.data.build_dataset` first")

    # unsloth must be imported before trl/transformers/peft to apply its patches.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTConfig, SFTTrainer

    compute_dtype = cfg["quantization"]["bnb_4bit_compute_dtype"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=cfg["max_seq_len"],
        dtype=None,  # auto: bfloat16 if supported else float16
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        random_state=cfg["seed"],
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        target_modules=cfg["lora"]["target_modules"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth" if cfg["gradient_checkpointing"] else False,
        random_state=cfg["seed"],
        max_seq_length=cfg["max_seq_len"],
    )

    label_space = label_space_from(pl.read_parquet(TRAIN_PARQUET))
    train_ds = load_split(TRAIN_PARQUET, label_space, tokenizer)
    val_ds = load_split(VAL_PARQUET, label_space, tokenizer) if VAL_PARQUET.exists() else None

    run_dir = ADAPTER_OUT / args.model_name.replace("/", "__")
    run_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(run_dir),
        per_device_train_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        num_train_epochs=cfg["num_train_epochs"],
        optim=cfg["optim"],
        seed=cfg["seed"],
        max_length=cfg["max_seq_len"],
        packing=cfg["packing"],
        dataset_text_field="text",
        bf16=(compute_dtype == "bfloat16"),
        fp16=(compute_dtype != "bfloat16"),
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=cfg["logging_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        eval_strategy="epoch" if val_ds is not None else "no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print(f"Training {args.model_name} on {len(train_ds)} contracts -> {run_dir}")
    trainer.train()

    model.save_pretrained(str(run_dir / "final_adapter"))
    tokenizer.save_pretrained(str(run_dir / "final_adapter"))
    print(f"Saved adapter to {run_dir / 'final_adapter'}")


if __name__ == "__main__":
    main()
