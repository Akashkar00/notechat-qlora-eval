# Fine-tune notebook GPU setup — resolved (2026-08-24)

Status: **working.** `notebooks/finetune.ipynb` runs natively on Windows with GPU
acceleration. A first training run was started and then deliberately stopped so the
external drive could be disconnected; no adapter was produced yet.

## Where the project lives now

`C:\Users\bmsip\notechat-qlora-eval` — moved off the external T7 SSD (`E:`) on
2026-08-24. The `E:` copy is stale from this point on; treat `C:` as authoritative.
Renamed from `company-finetune-eval` on 2026-08-28 to match the GitHub repo
(`Akashkar00/notechat-qlora-eval`); the `.venv` was rebuilt and the kernelspec
re-registered under the new name, for the same reason as the `E:` move below.

The `.venv` was **not** copied — it was rebuilt at the new location with
`uv sync --extra gpu` from uv's local cache. Copying a venv between paths leaves the
installed console-script `.exe` shims pointing at the old absolute path, and rebuilding
from cache is faster than copying 5.5GB over USB anyway.

## Kernel to select

VS Code → Select Kernel → **`Python 3.11 (notechat-qlora-eval GPU)`**, i.e.
`C:\Users\bmsip\notechat-qlora-eval\.venv\Scripts\python.exe` (Python 3.11.9, matching
`pyproject.toml`'s `requires-python = ">=3.11,<3.12"`).

Registered as a named kernelspec so it survives VS Code's environment-discovery quirks:

```
.venv\Scripts\python.exe -m ipykernel install --user \
  --name notechat-qlora-eval --display-name "Python 3.11 (notechat-qlora-eval GPU)"
```

Re-run that if the project ever moves again. `.vscode/settings.json` also pins
`python.defaultInterpreterPath` to the same interpreter.

Verified: `torch 2.11.0+cu128`, `torch.cuda.is_available() == True`, RTX 3060 (12GB,
sm_86), real GPU matmul, and unsloth / bitsandbytes / trl / peft / accelerate all import.

## Two non-obvious things that cost real time

### 1. PyPI serves CPU-only torch on Windows — silently

`pyproject.toml` originally declared a bare `torch>=2.4`. On win_amd64 that resolves to
`torch==2.11.0+cpu`, which installs with **no error or warning** and leaves
`torch.cuda.is_available() == False`. A fine-tune started in that state would have run on
CPU at unusable speed while looking fine.

Fixed by pinning torch/torchvision to PyTorch's CUDA index in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
```

Cell 1's `assert torch.cuda.is_available()` is what catches a regression here — don't
weaken it.

**The torch version was not a free choice.** `unsloth` requires `torch<2.12.0,>=2.4.0`
and `torchvision 0.26.0` pins `torch==2.11.0` exactly, so 2.11.0 is forced. The global
Python 3.11 install's `torch 2.13.0+cu126` could not have been reused for this reason.

**cu128 specifically**, out of the cu126/cu128/cu130 builds available for
torch 2.11.0/cp311/win: bitsandbytes 0.50.1 ships a matching
`libbitsandbytes_cuda128.dll`, and cu128 is the only one of the three with a
Windows/cp311 `xformers` wheel on the PyTorch index. (That index's xformers is only
0.0.30, which pairs with torch 2.7 — so xformers is left to come from PyPI at 0.0.35,
which requires `torch>=2.10` and is compatible.)

`torchvision` is listed explicitly in the `gpu` extra rather than left transitive via
unsloth, so the `[tool.uv.sources]` entry actually applies to it.

### 2. The E: drive filesystem was corrupt

Two installs failed with different impossible-looking errors (`os error 433: device does
not exist`, then `os error 1392: file or directory is corrupted and unreadable`). Root
cause was not antivirus or bad luck: `E:` (Samsung T7 Shield, **exFAT**) was flagged
`HealthStatus: Warning`, `OperationalStatus: Full Repair Needed`, and
`fsutil dirty query E:` reported the volume dirty.

`chkdsk E: /f /x` found and repaired corruption (concentrated in the half-written
`.venv\Lib\site-packages\torchao\...`); the volume then reported Healthy / not dirty. The
physical SSD was healthy throughout — this was filesystem-level damage, most likely from
an unsafe disconnect. exFAT has no journaling, so **eject that drive properly** if it is
used again.

## Notes / things not to redo

- **Don't reach for WSL2.** No distro is installed, "Virtual Machine Platform" is off
  (needs admin + reboot), and it is unnecessary — the full GPU stack installs and runs
  natively on Windows.
- **Don't try to reuse the `torch-gpu` conda env** (Python 3.12.11 — violates the
  project's `<3.12` pin) or the global Python 3.11 install. Neither has
  unsloth/bitsandbytes/trl/peft/accelerate, and the global one's torch 2.13 is above
  unsloth's cap.
- **Don't use `uv add` to add a package here.** Its implicit sync runs without
  `--extra gpu` and will uninstall the CUDA torch. Edit `pyproject.toml` directly, then
  run `uv sync --extra gpu`.

## Run configuration and observed behaviour

`MODEL_NAME` is set in the notebook to `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`, chosen at
build time per `PROJECT_SPEC.md` §7 (#1/#4) rather than assumed: it matches the
"~4B model on a 12GB card" sizing `configs/train.yaml` targets, and the card is Ampere
(sm_86) so the config's `bfloat16` compute dtype is supported. Pre-quantized, so the
download is ~2GB instead of ~6GB. `CHAT_TEMPLATE` stays `qwen-2.5`.

From the aborted run, for planning the real one:

- 8,000 train examples pack into 1,986 sequences → **250 steps** over 2 epochs
  (125 steps/epoch) at effective batch 16.
- ~**60 s/step** → roughly **4h10m–5h** end to end. Epoch 1 completes around the
  2h05m mark.
- VRAM at `max_seq_len=4096`: **8.8GB / 12GB**, GPU pinned at 100%, ~75°C. Comfortable
  headroom; no OOM. If a larger model is tried later, `configs/train.yaml` says reduce
  `max_seq_len` first, not LoRA rank.
- LoRA trains 29.9M of 3.12B params (0.96%).

**The progress table shows nothing until the end of epoch 1 (~2h in).** This is not a
stall. `transformers/utils/notebook.py:344` only writes a training-loss row when
`eval_strategy == "no"`; because this notebook uses `eval_strategy="epoch"`, rows are
written solely by `on_evaluate`, i.e. at epoch boundaries. Loss is still recorded to
`trainer.state.log_history` every 10 steps. Checkpoints also land only at epoch end
(`save_strategy="epoch"`), so an interrupt before ~2h loses the whole run.

## Known scope mismatch (unchanged, flagged in the notebook itself)

`docs/PROJECT_SPEC.md` §7, `docs/DECISIONS.md`, and `src/train/train.py` describe a CUAD
contract-clause-extraction task. The data actually built and on disk is **NoteChat
clinical dialogues** (`data/processed/{train,val}.parquet` — 8,000/1,000 rows, columns
`clinical_note` + `conversation`). The notebook is deliberately written against the data
that exists. Also note `docs/data_report.md`'s ground-truth caveat: the target dialogues
are themselves LLM-generated, so this fine-tune imitates another model's output and eval
numbers should be read in that light.
