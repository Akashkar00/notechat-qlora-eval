#!/usr/bin/env bash
# Phase 6 — run all four comparison arms end to end (PROJECT_SPEC.md §5 Phase 6).
#
# Sequential, not parallel: arms 1-3 each need most of a 12GB card, and
# arm 4's BERTScore pass needs the GPU too. Greedy decoding throughout, so
# re-running this script reproduces every number in README.md exactly.
set -euo pipefail

PY="${PY:-.venv/Scripts/python.exe}"
N="${N:-200}"
ADAPTER="artifacts/adapters/unsloth__Qwen2.5-3B-Instruct-bnb-4bit/final_adapter"

echo "=== Arm 1: zero-shot Qwen2.5-3B ==="
"$PY" -m src.eval.run_eval --arm zero-shot --n-records "$N"

echo "=== Arm 3: QLoRA fine-tuned Qwen2.5-3B ==="
"$PY" -m src.eval.run_eval --arm finetuned --n-records "$N" --adapter "$ADAPTER"

echo "=== Arm 2: zero-shot Qwen2.5-14B ==="
"$PY" -m src.eval.run_eval --arm zero-shot-14b --n-records "$N" \
    --model-name unsloth/Qwen2.5-14B-Instruct-bnb-4bit --max-seq-len 2048

echo "=== Arm 4: classic TF-IDF retrieval baseline ==="
"$PY" -m src.eval.run_eval --arm classic-tfidf --n-records "$N" --baseline tfidf

echo "=== Comparison across all arms ==="
"$PY" -m src.eval.compare

echo "=== ALL ARMS COMPLETE ==="
