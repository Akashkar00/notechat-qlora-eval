"""Local HF/unsloth inference runner (PROJECT_SPEC.md §3 src/inference/local_hf.py).

Mirrors notebooks/finetune.ipynb's prompt/generation cells exactly, so
run_eval.py doesn't duplicate that logic in a second place that can drift —
if the notebook's SYSTEM_PROMPT ever changes, update it here too (and vice
versa; src/train/train.py's rewrite, tracked in docs/DECISIONS.md, should
import from here rather than re-defining its own copy).
"""

SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. Given a clinical note, generate "
    "a realistic doctor-patient dialogue consistent with the note. Output only the "
    "dialogue, formatted as alternating `Doctor:`/`Patient:` turns — no other text."
)


def build_user_prompt(row: dict) -> str:
    return f"Clinical note:\n{row['clinical_note']}"


def load_model(
    model_name: str,
    max_seq_len: int,
    load_in_4bit: bool,
    seed: int,
    adapter_path: str | None = None,
    chat_template: str = "qwen-2.5",
):
    """Load a model for inference. Pass `adapter_path` to load a saved LoRA
    adapter on top of its base model (unsloth reads the base model name from
    the adapter's own adapter_config.json); omit it to get the frozen base
    model — this is what makes the zero-shot baseline arm (PROJECT_SPEC.md §5
    Phase 6 #1) and the fine-tuned arm (#3) the same code path.
    """
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path or model_name,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=load_in_4bit,
        random_state=seed,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate(
    model,
    tokenizer,
    note_row: dict,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(note_row)},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs.shape[1] :], skip_special_tokens=True)
