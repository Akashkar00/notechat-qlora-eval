"""Phase 5 — QLoRA fine-tuning (PROJECT_SPEC.md §5 Phase 5).

Task (PROJECT_SPEC.md §4): given a clinical note, generate a realistic
doctor-patient dialogue as alternating `Doctor:`/`Patient:` turns.

**Prompt logic is imported from `src/inference/local_hf.py`, not redefined
here.** Training and evaluation must agree on the system prompt and user
prompt exactly — if they drift, the fine-tune is optimised for one format
and scored on another, and the resulting numbers are quietly meaningless.
This module previously carried its own copy (of a different task's prompts,
no less) and drifted for days; importing is what stops that recurring.

Model + hyperparameters: configs/train.yaml. `model_name` is passed on the
command line rather than hardcoded, so the choice stays visible in the run
command (PROJECT_SPEC.md §5 — "confirm at build time, do not assume").

Run:
    python -m src.train.train --model-name unsloth/Qwen2.5-3B-Instruct-bnb-4bit
"""

import argparse
import json
from pathlib import Path

import polars as pl
import yaml

from src.inference.local_hf import SYSTEM_PROMPT, build_user_prompt

CONFIG_PATH = Path("configs/train.yaml")
TRAIN_PARQUET = Path("data/processed/train.parquet")
VAL_PARQUET = Path("data/processed/val.parquet")
ADAPTER_OUT = Path("artifacts/adapters")


def build_target(row: dict) -> str:
    """The assistant turn the model learns to produce: the dialogue itself.

    No JSON wrapper, no schema — this task's output is free-form text
    (contrast the CUAD version this file used to hold, which emitted a JSON
    array of clause/evidence pairs).
    """
    return row["conversation"]


def to_chat_text(row: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(row)},
        {"role": "assistant", "content": build_target(row)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def load_split(path: Path, tokenizer):
    from datasets import Dataset

    df = pl.read_parquet(path)
    texts = [to_chat_text(row, tokenizer) for row in df.iter_rows(named=True)]
    return Dataset.from_dict({"text": texts})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        required=True,
        help="HF checkpoint id, e.g. unsloth/Qwen2.5-3B-Instruct-bnb-4bit. Not defaulted — "
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

    train_ds = load_split(TRAIN_PARQUET, tokenizer)
    val_ds = load_split(VAL_PARQUET, tokenizer) if VAL_PARQUET.exists() else None

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

    print(f"Training {args.model_name} on {len(train_ds)} notes -> {run_dir}")
    trainer.train()

    # Explicit final evaluation. The notebook run of this same config left no
    # epoch-2 eval_loss in trainer_state.json (see docs/DECISIONS.md), which
    # made the final validation loss unrecoverable without retraining.
    # Calling evaluate() directly means the last number always exists,
    # regardless of how the callback-driven eval schedule behaves.
    final_metrics = {}
    if val_ds is not None:
        final_metrics = trainer.evaluate()
        print(f"Final validation metrics: {final_metrics}")

    final_dir = run_dir / "final_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Persist the loss curve next to the adapter so a training run's history
    # survives independently of the checkpoint dirs, which save_total_limit
    # prunes.
    (run_dir / "train_history.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "config": cfg,
                "final_eval": final_metrics,
                "log_history": trainer.state.log_history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved adapter to {final_dir}")


if __name__ == "__main__":
    main()
